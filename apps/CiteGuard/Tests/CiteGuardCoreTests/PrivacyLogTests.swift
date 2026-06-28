import XCTest
@testable import CiteGuardCore

final class PrivacyLogTests: XCTestCase {
    func testPrivacyLogRecordsOnlyCitationStrings() async {
        let log = PrivacyLogStore()
        await log.append(PrivacyLogEntry(mode: .live, jurisdictionHint: .hk, citationsSent: ["[2025] HKCFI 808"]))

        let entries = await log.allEntries()

        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries.first?.citationsSent, ["[2025] HKCFI 808"])
        XCTAssertTrue(entries.first?.note.localizedCaseInsensitiveContains("document text") == true)
    }
}
