import Foundation

public struct PrivacyLogEntry: Identifiable, Hashable, Codable, Sendable {
    public let id: UUID
    public let createdAt: Date
    public let mode: ResolutionMode
    public let jurisdictionHint: JurisdictionHint
    public let citationsSent: [String]
    public let note: String

    public init(
        id: UUID = UUID(),
        createdAt: Date = Date(),
        mode: ResolutionMode,
        jurisdictionHint: JurisdictionHint,
        citationsSent: [String],
        note: String = "Only citation strings were sent off-device. Document text and surrounding passages were not sent."
    ) {
        self.id = id
        self.createdAt = createdAt
        self.mode = mode
        self.jurisdictionHint = jurisdictionHint
        self.citationsSent = citationsSent
        self.note = note
    }
}

public actor PrivacyLogStore {
    private var entries: [PrivacyLogEntry] = []

    public init() {}

    public func append(_ entry: PrivacyLogEntry) {
        entries.insert(entry, at: 0)
    }

    public func allEntries() -> [PrivacyLogEntry] {
        entries
    }
}
