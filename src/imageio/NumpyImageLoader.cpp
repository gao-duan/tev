// This file was developed by Duan Gao <gao-d17@mails.tsinghua.edu.cn>.
// It is published under the BSD 3-Clause License within the LICENSE file.

#include <tev/imageio/NumpyImageLoader.h>
#include <tev/ThreadPool.h>


using namespace Eigen;
using namespace filesystem;
using namespace std;

TEV_NAMESPACE_BEGIN

bool NumpyImageLoader::canLoadFile(std::istream& iStream) const
{
    std::string header;
    std::getline(iStream, header);

    bool result = !!iStream && header.length() >= 10 && header.substr(1, 5) == "NUMPY" && (unsigned char)header[0] == (unsigned char)0x93;

    iStream.clear();
    iStream.seekg(0);

    return result;
}

ImageData NumpyImageLoader::load(std::istream& iStream, const filesystem::path& path, const std::string& channelSelector) const
{
    ImageData result;


    // parse npy header
    std::string header;
    std::getline(iStream, header);

    if (header.length() < 10 || header.substr(1, 5) != "NUMPY" || (unsigned char)header[0] != (unsigned char)0x93) {
        throw invalid_argument{ tfm::format("Invalid numpy image") };
    }

    /*
    std::ifstream file(fn.c_str(), std::ios::binary);
                if (!file) return false;

                std::string header;
                std::getline(file, header);
                if (header.length() < 10 || header.substr(1, 5) != "NUMPY")
                    return false;

                // Format
                auto pos = header.find("descr");
                if (pos == std::string::npos)
                    return false;
                pos += 9;
                bool littleEndian = (header[pos] == '<' || header[pos] == '|' ? true : false);
                if (!littleEndian)
                    return false; // Only supports little endian.
                char type = header[pos + 1];
                int size = atoi(header.substr(pos + 2, 1).c_str()); // assume size <= 8 bytes
                if (type != 'f' && type != 'u' && size > 4)
                    return false;

                // Order
                pos = header.find("fortran_order");
                if (pos == std::string::npos)
                    return false;
                pos += 16;
                bool fortranOrder = header.substr(pos, 4) == "True" ? true : false;

                if (fortranOrder) // Only supports C order.
                    return false;

                // Shape
                auto offset = header.find("(") + 1;
                auto shapeString = header.substr(offset, header.find(")") - offset);
                std::regex regex("[0-9][0-9]*");
                std::smatch match;
                std::vector<int> shape;
                while (std::regex_search(shapeString, match, regex))
                {
                    shape.push_back(std::stoi(match[0].str()));
                    shapeString = match.suffix().str();
                }
                int w = 0, h = 0, ch = 1;
                if (shape.size() < 2 || shape.size() > 4) // support 2/3/4
                {
                    return false;
                }
                if (shape.size() == 2)
                {
                    h = shape[0];
                    w = shape[1];
                    ch = 1;
                }
                else if (shape.size() == 3)
                {
                    h = shape[0];
                    w = shape[1];
                    ch = shape[2];
                }
                else if (shape.size() == 4)
                {
                    //if (shape[0] > 1) // single image only
                    //	return false;
                    h = shape[1];
                    w = shape[2];
                    ch = shape[3];

                }
                if (ch > 4) // at most 4 channel
                    return false;

                std::vector<char> data;
                data.resize(w * h * ch * size);
                if (!file.read(data.data(), w * h * ch * size))
                    return false;

                if (!dstImg.Create(w, h))
                    return false;

                if (type == 'f' && size == 2)
                {
                    if (ch == 1)
                    {
                        ig::CImage_1c16f tmpImg;
                        if (!tmpImg.Create(w, h, w * ch * size, data.data()))
                            return false;
                        ig::ConvertImageType(tmpImg, dstImg);
                    }
                    else if (ch == 2)
                    {
                        ig::CImage_2c16f tmpImg;
                        if (!tmpImg.Create(w, h, w * ch * size, data.data()))
                            return false;
                        ig::ConvertImageType(tmpImg, dstImg);
                    }
                    else if (ch == 3)
                    {
                        ig::CImage_3c16f tmpImg;
                        if (!tmpImg.Create(w, h, w * ch * size, data.data()))
                            return false;
                        ig::ConvertImageType(tmpImg, dstImg);
                    }
                    else if (ch == 4)
                    {
                        ig::CImage_4c16f tmpImg;
                        if (!tmpImg.Create(w, h, w * ch * size, data.data()))
                            return false;
                        ig::ConvertImageType(tmpImg, dstImg);
                    }
                }
                else if (type == 'f' && size == 4)
                {
                    if (ch == 1)
                    {
                        ig::CImage_1c32f tmpImg;
                        if (!tmpImg.Create(w, h, w * ch * size, data.data()))
                            return false;
                        ig::ConvertImageType(tmpImg, dstImg);
                    }
                    else if (ch == 2)
                    {
                        ig::CImage_2c32f tmpImg;
                        if (!tmpImg.Create(w, h, w * ch * size, data.data()))
                            return false;
                        ig::ConvertImageType(tmpImg, dstImg);
                    }
                    else if (ch == 3)
                    {
                        ig::CImage_3c32f tmpImg;
                        if (!tmpImg.Create(w, h, w * ch * size, data.data()))
                            return false;
                        ig::ConvertImageType(tmpImg, dstImg);
                    }
                    else if (ch == 4)
                    {
                        ig::CImage_4c32f tmpImg;
                        if (!tmpImg.Create(w, h, w * ch * size, data.data()))
                            return false;
                        ig::ConvertImageType(tmpImg, dstImg);
                    }
                }
                else if (type == 'u' && size == 1)
                {
                    if (ch == 1)
                    {
                        ig::CImage_1c8u tmpImg;
                        if (!tmpImg.Create(w, h, w * ch * size, data.data()))
                            return false;
                        ig::ConvertImageType(tmpImg, dstImg);
                    }
                    else if (ch == 2)
                    {
                        ig::CImage_2c8u tmpImg;
                        if (!tmpImg.Create(w, h, w * ch * size, data.data()))
                            return false;
                        ig::ConvertImageType(tmpImg, dstImg);
                    }
                    else if (ch == 3)
                    {
                        ig::CImage_3c8u tmpImg;
                        if (!tmpImg.Create(w, h, w * ch * size, data.data()))
                            return false;
                        ig::ConvertImageType(tmpImg, dstImg);
                    }
                    else if (ch == 4)
                    {
                        ig::CImage_4c8u tmpImg;
                        if (!tmpImg.Create(w, h, w * ch * size, data.data()))
                            return false;
                        ig::ConvertImageType(tmpImg, dstImg);
                    }
                }
                else
                {
                    return false; // not supported.
                }
    */



    return ImageData();
}

TEV_NAMESPACE_END
