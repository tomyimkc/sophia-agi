#if canImport(FoundationXML)
import Foundation
import FoundationXML
import zlib

final class DOCXTextExtractor: NSObject, XMLParserDelegate {
    private var textFragments: [String] = []
    private var currentText: String = ""
    private var insideTextNode = false

    static func extract(url: URL) throws -> String {
        let archive = try Data(contentsOf: url)
        let data = try ZIPDocumentXMLReader.documentXML(from: archive)
        guard !data.isEmpty else {
            throw DocumentIntakeError.unreadableDocument
        }

        let delegate = DOCXTextExtractor()
        let parser = XMLParser(data: data)
        parser.delegate = delegate
        guard parser.parse() else {
            throw DocumentIntakeError.unreadableDocument
        }
        return delegate.textFragments.joined(separator: " ")
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
    }

    func parser(_ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?, qualifiedName qName: String?, attributes attributeDict: [String: String] = [:]) {
        if elementName == "w:t" || elementName == "t" {
            insideTextNode = true
            currentText = ""
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        if insideTextNode {
            currentText += string
        }
    }

    func parser(_ parser: XMLParser, didEndElement elementName: String, namespaceURI: String?, qualifiedName qName: String?) {
        if elementName == "w:t" || elementName == "t" {
            insideTextNode = false
            if !currentText.isEmpty {
                textFragments.append(currentText)
            }
            currentText = ""
        }
    }
}

private enum ZIPDocumentXMLReader {
    private static let endOfCentralDirectorySignature: UInt32 = 0x06054b50
    private static let centralDirectorySignature: UInt32 = 0x02014b50
    private static let localHeaderSignature: UInt32 = 0x04034b50

    static func documentXML(from archive: Data) throws -> Data {
        guard let eocdOffset = findEndOfCentralDirectory(in: archive) else {
            throw DocumentIntakeError.unreadableDocument
        }
        let entryCount = Int(archive.uint16(at: eocdOffset + 10))
        let centralDirectoryOffset = Int(archive.uint32(at: eocdOffset + 16))

        var cursor = centralDirectoryOffset
        for _ in 0..<entryCount {
            guard archive.uint32(at: cursor) == centralDirectorySignature else {
                throw DocumentIntakeError.unreadableDocument
            }

            let method = archive.uint16(at: cursor + 10)
            let compressedSize = Int(archive.uint32(at: cursor + 20))
            let uncompressedSize = Int(archive.uint32(at: cursor + 24))
            let nameLength = Int(archive.uint16(at: cursor + 28))
            let extraLength = Int(archive.uint16(at: cursor + 30))
            let commentLength = Int(archive.uint16(at: cursor + 32))
            let localHeaderOffset = Int(archive.uint32(at: cursor + 42))
            let nameRange = cursor + 46..<(cursor + 46 + nameLength)
            guard nameRange.upperBound <= archive.count else {
                throw DocumentIntakeError.unreadableDocument
            }
            let name = String(data: archive.subdata(in: nameRange), encoding: .utf8)

            if name == "word/document.xml" {
                return try readLocalEntry(
                    archive: archive,
                    offset: localHeaderOffset,
                    method: method,
                    compressedSize: compressedSize,
                    uncompressedSize: uncompressedSize
                )
            }

            cursor += 46 + nameLength + extraLength + commentLength
        }

        throw DocumentIntakeError.unreadableDocument
    }

    private static func readLocalEntry(
        archive: Data,
        offset: Int,
        method: UInt16,
        compressedSize: Int,
        uncompressedSize: Int
    ) throws -> Data {
        guard archive.uint32(at: offset) == localHeaderSignature else {
            throw DocumentIntakeError.unreadableDocument
        }

        let nameLength = Int(archive.uint16(at: offset + 26))
        let extraLength = Int(archive.uint16(at: offset + 28))
        let dataStart = offset + 30 + nameLength + extraLength
        let dataEnd = dataStart + compressedSize
        guard dataStart >= 0, dataEnd <= archive.count else {
            throw DocumentIntakeError.unreadableDocument
        }

        let compressed = archive.subdata(in: dataStart..<dataEnd)
        switch method {
        case 0:
            return compressed
        case 8:
            return try inflateRawDeflate(compressed, expectedSize: uncompressedSize)
        default:
            throw DocumentIntakeError.unreadableDocument
        }
    }

    private static func inflateRawDeflate(_ data: Data, expectedSize: Int) throws -> Data {
        var stream = z_stream()
        let initStatus = inflateInit2_(&stream, -MAX_WBITS, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size))
        guard initStatus == Z_OK else {
            throw DocumentIntakeError.unreadableDocument
        }
        defer { inflateEnd(&stream) }

        var output = Data(count: max(expectedSize, 4096))
        let status: Int32 = data.withUnsafeBytes { inputBuffer in
            guard let inputBase = inputBuffer.bindMemory(to: Bytef.self).baseAddress else {
                return Z_DATA_ERROR
            }
            stream.next_in = UnsafeMutablePointer(mutating: inputBase)
            stream.avail_in = uInt(data.count)

            return output.withUnsafeMutableBytes { outputBuffer in
                guard let outputBase = outputBuffer.bindMemory(to: Bytef.self).baseAddress else {
                    return Z_DATA_ERROR
                }
                stream.next_out = outputBase
                stream.avail_out = uInt(output.count)
                return inflate(&stream, Z_FINISH)
            }
        }

        guard status == Z_STREAM_END else {
            throw DocumentIntakeError.unreadableDocument
        }

        output.count = Int(stream.total_out)
        return output
    }

    private static func findEndOfCentralDirectory(in data: Data) -> Int? {
        guard data.count >= 22 else { return nil }
        let lowerBound = max(0, data.count - 65_557)
        var offset = data.count - 22
        while offset >= lowerBound {
            if data.uint32(at: offset) == endOfCentralDirectorySignature {
                return offset
            }
            offset -= 1
        }
        return nil
    }
}

private extension Data {
    func uint16(at offset: Int) -> UInt16 {
        guard offset + 2 <= count else { return 0 }
        return UInt16(self[offset]) | (UInt16(self[offset + 1]) << 8)
    }

    func uint32(at offset: Int) -> UInt32 {
        guard offset + 4 <= count else { return 0 }
        return UInt32(self[offset])
            | (UInt32(self[offset + 1]) << 8)
            | (UInt32(self[offset + 2]) << 16)
            | (UInt32(self[offset + 3]) << 24)
    }
}
#endif
