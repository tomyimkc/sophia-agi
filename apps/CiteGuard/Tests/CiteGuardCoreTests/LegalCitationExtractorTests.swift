import XCTest
@testable import CiteGuardCore

final class LegalCitationExtractorTests: XCTestCase {
    func testExtractsAndNormalizesHongKongCitations() {
        let extractor = HKCitationExtractor()
        let citations = extractor.extract(from: "See [2025]  hkcfi 808 and Cap 614 before filing.")

        XCTAssertEqual(citations.map(\.normalized), ["[2025] HKCFI 808", "Cap. 614"])
        XCTAssertTrue(citations.allSatisfy(\.isSupportedByLaunchScope))
    }

    func testFabricatedHongKongCitationIsStillExtractedForFailClosedResolution() {
        let extractor = HKCitationExtractor()
        let citations = extractor.extract(from: "As established in Wong v Lee [2025] HKCFI 9999, the point is settled.")

        XCTAssertEqual(citations.first?.normalized, "[2025] HKCFI 9999")
        XCTAssertEqual(citations.first?.jurisdiction, .hk)
        XCTAssertEqual(citations.first?.courtToken, "HKCFI")
        XCTAssertEqual(citations.first?.isSupportedByLaunchScope, true)
    }

    func testUSMataStyleReporterCitationIsUnsupportedInHKLaunchScope() {
        let extractor = HKCitationExtractor()
        let citations = extractor.extract(from: "See Varghese v. China Southern Airlines, 925 F.3d 1339 (11th Cir. 2019).")

        XCTAssertEqual(citations.first?.normalized, "925 F.3d 1339")
        XCTAssertEqual(citations.first?.jurisdiction, .us)
        XCTAssertEqual(citations.first?.isSupportedByLaunchScope, false)
    }

    func testDoesNotTreatOrdinaryNumbersAsCitations() {
        let extractor = HKCitationExtractor()
        XCTAssertTrue(extractor.extract(from: "We met 12 of 30 criteria.").isEmpty)
    }
}
