import Foundation

#if canImport(PDFKit)
import PDFKit
#endif

public enum DocumentIntakeError: LocalizedError, Equatable {
    case missingFile
    case unsupportedFileType(String)
    case unreadableDocument

    public var errorDescription: String? {
        switch self {
        case .missingFile:
            return "File not found."
        case .unsupportedFileType(let ext):
            return "Unsupported file type: \(ext). CiteGuard v1 accepts PDF, DOCX, TXT, MD, and pasted text."
        case .unreadableDocument:
            return "Could not read document text on-device."
        }
    }
}

public struct DocumentIntake: Sendable {
    public init() {}

    public func extractText(from url: URL) throws -> String {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw DocumentIntakeError.missingFile
        }

        switch url.pathExtension.lowercased() {
        case "txt", "md":
            return try String(contentsOf: url, encoding: .utf8)
        case "docx":
            return try extractDOCXText(from: url)
        case "pdf":
            return try extractPDFText(from: url)
        default:
            throw DocumentIntakeError.unsupportedFileType(url.pathExtension.lowercased())
        }
    }

    private func extractDOCXText(from url: URL) throws -> String {
        #if canImport(FoundationXML)
        return try DOCXTextExtractor.extract(url: url)
        #else
        throw DocumentIntakeError.unreadableDocument
        #endif
    }

    private func extractPDFText(from url: URL) throws -> String {
        #if canImport(PDFKit)
        guard let document = PDFDocument(url: url) else {
            throw DocumentIntakeError.unreadableDocument
        }
        var pages: [String] = []
        for index in 0..<document.pageCount {
            if let text = document.page(at: index)?.string {
                pages.append(text)
            }
        }
        return pages.joined(separator: "\n")
        #else
        throw DocumentIntakeError.unreadableDocument
        #endif
    }
}
