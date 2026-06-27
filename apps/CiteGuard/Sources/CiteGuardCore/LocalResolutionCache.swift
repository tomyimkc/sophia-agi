import Foundation

public actor LocalResolutionCache {
    private var storage: [String: CitationResolution] = [:]

    public init() {}

    public func value(for normalizedCitation: String, source: String? = nil) -> CitationResolution? {
        storage[key(normalizedCitation, source: source)]
    }

    public func store(_ resolution: CitationResolution) {
        storage[key(resolution.normalized, source: resolution.source)] = resolution
    }

    private func key(_ normalizedCitation: String, source: String?) -> String {
        "\(normalizedCitation.lowercased())|\(source ?? "any")"
    }
}
