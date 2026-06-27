import Foundation

public protocol CitationResolving: Sendable {
    func resolve(
        candidates: [CitationCandidate],
        mode: ResolutionMode,
        jurisdictionHint: JurisdictionHint
    ) async -> [CitationResolution]
}

public struct CiteGuardAPIClient: CitationResolving {
    public enum ClientError: Error {
        case missingAPIKey
    }

    private struct ResolveRequest: Encodable {
        let citations: [String]
        let mode: ResolutionMode
        let jurisdictionHint: JurisdictionHint

        enum CodingKeys: String, CodingKey {
            case citations
            case mode
            case jurisdictionHint = "jurisdiction_hint"
        }
    }

    private struct ResolveResponse: Decodable {
        let apiVersion: String
        let mode: ResolutionMode
        let results: [APIResult]

        enum CodingKeys: String, CodingKey {
            case apiVersion = "api_version"
            case mode
            case results
        }
    }

    private struct APIResult: Decodable {
        let citation: String
        let normalized: String
        let courtToken: String?
        let state: CitationState
        let source: String?
        let sourceURL: URL?
        let retrievedAt: Date?
        let reason: String?

        enum CodingKeys: String, CodingKey {
            case citation
            case normalized
            case courtToken = "court_token"
            case state
            case source
            case sourceURL = "source_url"
            case retrievedAt = "retrieved_at"
            case reason
        }

        func toResolution(fallback: CitationCandidate) -> CitationResolution {
            CitationResolution(
                citation: citation,
                normalized: normalized,
                courtToken: courtToken,
                state: state,
                source: source,
                sourceURL: sourceURL,
                retrievedAt: retrievedAt,
                reason: reason ?? defaultReason(for: state)
            )
        }

        private func defaultReason(for state: CitationState) -> String {
            switch state {
            case .confirmed:
                return "Found in the named source. This confirms citation existence only."
            case .couldNotConfirm:
                return "Could not confirm against covered sources. Stop and check the official source yourself."
            case .unsupported:
                return "Citation jurisdiction or format is outside current CiteGuard coverage."
            }
        }
    }

    private let baseURL: URL
    private let apiKey: String?
    private let session: URLSession
    private let timeout: TimeInterval
    private let localSnapshot: BundledAuthoritySnapshot
    private let cache: LocalResolutionCache
    private let privacyLog: PrivacyLogStore?

    public init(
        baseURL: URL = URL(string: "https://api.citeguard.example/v1")!,
        apiKey: String? = nil,
        session: URLSession = .shared,
        timeout: TimeInterval = 12,
        localSnapshot: BundledAuthoritySnapshot = BundledAuthoritySnapshot(),
        cache: LocalResolutionCache = LocalResolutionCache(),
        privacyLog: PrivacyLogStore? = nil
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.session = session
        self.timeout = timeout
        self.localSnapshot = localSnapshot
        self.cache = cache
        self.privacyLog = privacyLog
    }

    public func resolve(
        candidates: [CitationCandidate],
        mode: ResolutionMode,
        jurisdictionHint: JurisdictionHint
    ) async -> [CitationResolution] {
        guard !candidates.isEmpty else { return [] }

        var finalResults: [CitationResolution] = []
        var networkCandidates: [CitationCandidate] = []

        for candidate in candidates {
            if !candidate.isSupportedByLaunchScope {
                finalResults.append(localSnapshot.resolve(candidate))
            } else if mode == .cache {
                finalResults.append(localSnapshot.resolve(candidate))
            } else if let cached = await cache.value(for: candidate.normalized) {
                finalResults.append(cached)
            } else {
                networkCandidates.append(candidate)
            }
        }

        if !networkCandidates.isEmpty {
            let resolved = await resolveLive(networkCandidates, jurisdictionHint: jurisdictionHint)
            for item in resolved {
                if item.state == .confirmed {
                    await cache.store(item)
                }
            }
            finalResults.append(contentsOf: resolved)
        }

        let order = Dictionary(uniqueKeysWithValues: candidates.enumerated().map { ($0.element.normalized, $0.offset) })
        return finalResults.sorted { lhs, rhs in
            (order[lhs.normalized] ?? Int.max) < (order[rhs.normalized] ?? Int.max)
        }
    }

    private func resolveLive(_ candidates: [CitationCandidate], jurisdictionHint: JurisdictionHint) async -> [CitationResolution] {
        guard let apiKey, !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return candidates.map { failClosed($0, reason: "No API key configured. Could not contact live sources; stop and check this yourself or use offline snapshot mode.") }
        }

        let citations = candidates.map { String($0.normalized.prefix(120)) }
        await privacyLog?.append(PrivacyLogEntry(mode: .live, jurisdictionHint: jurisdictionHint, citationsSent: citations))

        do {
            var request = URLRequest(url: baseURL.appendingPathComponent("resolve"))
            request.httpMethod = "POST"
            request.timeoutInterval = timeout
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            request.httpBody = try JSONEncoder().encode(ResolveRequest(citations: citations, mode: .live, jurisdictionHint: jurisdictionHint))

            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                return candidates.map { failClosed($0, reason: "Resolution service was unreachable or returned an error. Could not confirm; stop and check the official source yourself.") }
            }

            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let decoded = try decoder.decode(ResolveResponse.self, from: data)
            let byNormalized = Dictionary(uniqueKeysWithValues: decoded.results.map { ($0.normalized, $0) })

            return candidates.map { candidate in
                guard let apiResult = byNormalized[candidate.normalized] else {
                    return failClosed(candidate, reason: "Resolution response did not include this citation. Could not confirm; stop and check it yourself.")
                }
                return apiResult.toResolution(fallback: candidate)
            }
        } catch {
            return candidates.map { failClosed($0, reason: "Resolution request failed. Could not confirm; stop and check the official source yourself.") }
        }
    }

    private func failClosed(_ candidate: CitationCandidate, reason: String) -> CitationResolution {
        CitationResolution(
            citation: candidate.raw,
            normalized: candidate.normalized,
            courtToken: candidate.courtToken,
            state: .couldNotConfirm,
            source: nil,
            sourceURL: nil,
            retrievedAt: Date(),
            reason: reason
        )
    }
}
