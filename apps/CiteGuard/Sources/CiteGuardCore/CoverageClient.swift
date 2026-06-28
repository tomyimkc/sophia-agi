import Foundation

public protocol CoverageFetching: Sendable {
    func fetchCoverage() async -> [CoverageSource]
}

public struct CiteGuardCoverageClient: CoverageFetching {
    private struct CoverageResponse: Decodable {
        let sources: [Source]
    }

    private struct Source: Decodable {
        let jurisdiction: String
        let source: String
        let completenessNote: String
        let lastSnapshotDate: String?

        enum CodingKeys: String, CodingKey {
            case jurisdiction
            case source
            case completenessNote = "completeness_note"
            case lastSnapshotDate = "last_snapshot_date"
        }

        func toCoverageSource() -> CoverageSource {
            CoverageSource(
                jurisdiction: jurisdiction,
                source: source,
                completenessNote: completenessNote,
                lastSnapshotDate: lastSnapshotDate
            )
        }
    }

    private let baseURL: URL
    private let session: URLSession
    private let fallback: BundledAuthoritySnapshot

    public init(
        baseURL: URL = URL(string: "https://api.citeguard.example/v1")!,
        session: URLSession = .shared,
        fallback: BundledAuthoritySnapshot = BundledAuthoritySnapshot()
    ) {
        self.baseURL = baseURL
        self.session = session
        self.fallback = fallback
    }

    public func fetchCoverage() async -> [CoverageSource] {
        do {
            let (data, response) = try await session.data(from: baseURL.appendingPathComponent("coverage"))
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                return [fallback.coverage]
            }
            let decoded = try JSONDecoder().decode(CoverageResponse.self, from: data)
            return decoded.sources.map { $0.toCoverageSource() }
        } catch {
            return [fallback.coverage]
        }
    }
}
