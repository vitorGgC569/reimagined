#ifndef OX3_SERIALIZER_H
#define OX3_SERIALIZER_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>

// OxtaCore V3 Format Serializer (C++)
// Header: [Magic: 'OXH3'] [Version: u16] [Flags: u16]
// Record: [Marker: 0xDEADBEEF] [Score: f32] [Size: u32] [Payload: DeltaEncodedTokens]

class OX3Serializer {
public:
    OX3Serializer(const std::string& filename) {
        outfile.open(filename, std::ios::binary);
        if (!outfile.is_open()) {
            std::cerr << "[OX3] Error opening file: " << filename << std::endl;
            return;
        }

        // Write Header
        const char magic[4] = {'O', 'X', 'H', '3'};
        uint16_t version = 3;
        uint16_t flags = 0;
        outfile.write(magic, 4);
        outfile.write(reinterpret_cast<const char*>(&version), 2);
        outfile.write(reinterpret_cast<const char*>(&flags), 2);
    }

    ~OX3Serializer() {
        if (outfile.is_open()) outfile.close();
    }

    void write_record(const std::vector<int>& tokens, float score) {
        if (!outfile.is_open()) return;

        // Delta Encoding Buffer
        std::vector<uint8_t> buffer;

        // Token Count
        uint32_t count = tokens.size();
        append_u32(buffer, count);

        if (count > 0) {
            // Base Token (First)
            int last_val = tokens[0];
            append_u32(buffer, (uint32_t)last_val);

            // Deltas
            for (size_t i = 1; i < count; ++i) {
                int val = tokens[i];
                int delta = val - last_val;
                last_val = val;

                // Simple Delta Packing (matches Python logic)
                if (delta >= -32768 && delta <= 32767) {
                    if (delta == -32768) {
                        append_u16(buffer, 0x8000); // Escape
                        append_i32(buffer, delta);
                    } else {
                        append_i16(buffer, (int16_t)delta);
                    }
                } else {
                    append_u16(buffer, 0x8000); // Escape
                    append_i32(buffer, delta);
                }
            }
        }

        // Write Record Structure
        uint32_t marker = 0xDEADBEEF;
        uint32_t size = buffer.size();

        outfile.write(reinterpret_cast<const char*>(&marker), 4);
        outfile.write(reinterpret_cast<const char*>(&score), 4);
        outfile.write(reinterpret_cast<const char*>(&size), 4);
        outfile.write(reinterpret_cast<const char*>(buffer.data()), size);
    }

private:
    std::ofstream outfile;

    void append_u32(std::vector<uint8_t>& buf, uint32_t val) {
        uint8_t* p = reinterpret_cast<uint8_t*>(&val);
        for(int i=0; i<4; ++i) buf.push_back(p[i]);
    }
    void append_i32(std::vector<uint8_t>& buf, int32_t val) {
        uint8_t* p = reinterpret_cast<uint8_t*>(&val);
        for(int i=0; i<4; ++i) buf.push_back(p[i]);
    }
    void append_u16(std::vector<uint8_t>& buf, uint16_t val) {
        uint8_t* p = reinterpret_cast<uint8_t*>(&val);
        for(int i=0; i<2; ++i) buf.push_back(p[i]);
    }
    void append_i16(std::vector<uint8_t>& buf, int16_t val) {
        uint8_t* p = reinterpret_cast<uint8_t*>(&val);
        for(int i=0; i<2; ++i) buf.push_back(p[i]);
    }
};

#endif
