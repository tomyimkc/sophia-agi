import Foundation

public struct CitationScanner: Sendable {
    private let extractor: any CitationExtracting
    private let resolver: any CitationResolving
    private let jurisdictionHint: JurisdictionHint

    public init(
        extractor: any CitationExtracting = HKCitationExtractor(),
        resolver: any CitationResolving,
        jurisdictionHint: JurisdictionHint = .hk
    ) {
        self.extractor = extractor
        self.resolver = resolver
        self.jurisdictionHint = jurisdictionHint
    }

    public func scan(text: String, mode: ResolutionMode) async -> [CitationResolution] {
        let candidates = extractor.extract(from: text)
        return await resolver.resolve(candidates: candidates, mode: mode, jurisdictionHint: jurisdictionHint)
    }
}
