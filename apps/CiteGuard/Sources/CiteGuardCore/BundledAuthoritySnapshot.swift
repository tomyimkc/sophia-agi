import Foundation

public struct BundledAuthoritySnapshot: Sendable {
    private struct SnapshotFile: Decodable {
        let completenessNote: String
        let authorities: [Authority]

        enum CodingKeys: String, CodingKey {
            case completenessNote = "completeness_note"
            case authorities
        }
    }

    private struct Authority: Decodable {
        let citation: String
        let name: String
        let source: String
        let sourceURL: URL?
        let retrievedAt: Date?

        enum CodingKeys: String, CodingKey {
            case citation
            case name
            case source
            case sourceURL = "source_url"
            case retrievedAt = "retrieved_at"
        }
    }

    private let authorities: [String: Authority]
    public let coverage: CoverageSource

    public init() {
        self.init(bundle: .module)
    }

    init(bundle: Bundle) {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        if let url = bundle.url(forResource: "hk_authorities_snapshot", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let file = try? decoder.decode(SnapshotFile.self, from: data) {
            var mapped: [String: Authority] = [:]
            let extractor = HKCitationExtractor()
            for authority in file.authorities {
                mapped[extractor.normalize(authority.citation)] = authority
            }
            authorities = mapped
            coverage = CoverageSource(
                jurisdiction: "HK",
                source: "Bundled HK snapshot",
                completenessNote: file.completenessNote,
                lastSnapshotDate: "2026-06-22"
            )
        } else {
            authorities = [:]
            coverage = CoverageSource(
                jurisdiction: "HK",
                source: "Bundled HK snapshot",
                completenessNote: "Snapshot unavailable. Offline mode cannot confirm citations and will fail closed."
            )
        }
    }

    public func resolve(_ candidate: CitationCandidate) -> CitationResolution {
        guard candidate.isSupportedByLaunchScope else {
            return unsupported(candidate)
        }

        guard let authority = authorities[candidate.normalized] else {
            return CitationResolution(
                citation: candidate.raw,
                normalized: candidate.normalized,
                courtToken: candidate.courtToken,
                state: .couldNotConfirm,
                source: coverage.source,
                sourceURL: nil,
                retrievedAt: Date(),
                reason: "Not found in the bundled Hong Kong snapshot. This may be outside or newer than snapshot coverage; stop and check the official source yourself."
            )
        }

        return CitationResolution(
            citation: candidate.raw,
            normalized: candidate.normalized,
            courtToken: candidate.courtToken,
            state: .confirmed,
            source: authority.source,
            sourceURL: authority.sourceURL,
            retrievedAt: authority.retrievedAt,
            reason: "Found as \(authority.name). This confirms citation existence only."
        )
    }

    private func unsupported(_ candidate: CitationCandidate) -> CitationResolution {
        CitationResolution(
            citation: candidate.raw,
            normalized: candidate.normalized,
            courtToken: candidate.courtToken,
            state: .unsupported,
            source: nil,
            sourceURL: nil,
            retrievedAt: nil,
            reason: "CiteGuard v1 is configured for Hong Kong citations only. This citation was not checked."
        )
    }
}
