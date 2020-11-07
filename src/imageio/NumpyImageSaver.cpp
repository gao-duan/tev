// This file was developed by Duan Gao <gao-d17@mails.tsinghua.edu.cn>.
// It is published under the BSD 3-Clause License within the LICENSE file.

#include <tev/imageio/NumpyImageSaver.h>

using namespace Eigen;
using namespace filesystem;
using namespace std;

TEV_NAMESPACE_BEGIN

char BigEndianTest() {
    int x = 1;
    return (((char*)& x)[0]) ? '<' : '>';
}

void append_chars(std::vector<char>& buffer, const char* str) {
    buffer.insert(buffer.end(), str, str + strlen(str));
}

void append_chars(std::vector<char>& buffer, const std::string& str) {
    buffer.insert(buffer.end(), str.begin(), str.end());
}

void append_chars(std::vector<char>& buffer, const std::vector<char>& str) {
    buffer.insert(buffer.end(), str.begin(), str.end());
}

void append_chars(std::vector<char>& buffer, char c) {
    buffer.push_back(c);
}


void NumpyImageSaver::save(std::ostream& oStream, const filesystem::path& path, const std::vector<float>& data, const Eigen::Vector2i& imageSize, int nChannels) const
{
    std::vector<char> header;

    // descr of header
    std::vector<char> descr;
    append_chars(descr, "{'descr': '");
    append_chars(descr, BigEndianTest());
    append_chars(descr, 'f');
    append_chars(descr, std::to_string(sizeof(float)));
    append_chars(descr, "', 'fortran_order': False, 'shape': (");
    append_chars(descr, std::to_string(imageSize.x()));
    append_chars(descr, ',');
    append_chars(descr, std::to_string(imageSize.y()));
    append_chars(descr, ',');
    append_chars(descr, std::to_string(nChannels));
    append_chars(descr, "),}");

    //pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
    int remainder = 16 - (10 + descr.size()) % 16;
    append_chars(descr, std::string(' ', remainder));
    descr.back() = '\n';
   
    // header
    append_chars(header, (char)0x93);
    append_chars(header, "NUMPY");
    append_chars(header, (char)0x01); //major version of numpy format
    append_chars(header, (char)0x00);  //minor version of numpy format
    uint16_t header_size = (uint16_t)descr.size();
    append_chars(header, (header_size >> 8) & 0xff );
    append_chars(header, header_size & 0xff);
    append_chars(header, descr);

    // data
    const char* byte_data = reinterpret_cast<const char*>(data.data());
    assert(data.size() == imageSize.x() * imageSize.y() * nChannels);

    oStream.write(header.data(), header.size());
    oStream.write(byte_data, data.size() * sizeof(float));

}

TEV_NAMESPACE_END
