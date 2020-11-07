// This file was developed by Duan Gao <gao-d17@mails.tsinghua.edu.cn>.
// It is published under the BSD 3-Clause License within the LICENSE file.

#include <tev/imageio/NumpyImageLoader.h>
#include <tev/ThreadPool.h>
#include <regex>

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
    ThreadPool threadPool;

    Vector2i img_size;

    // parse npy header
    std::string header;
    std::getline(iStream, header);

    if (header.length() < 10 || header.substr(1, 5) != "NUMPY" || (unsigned char)header[0] != (unsigned char)0x93) {
        throw invalid_argument{ tfm::format("Invalid numpy image") };
    }


    // Format
    auto pos = header.find("descr");
    if (pos == std::string::npos) {
        throw invalid_argument{ tfm::format("Numpy image cannot find `descr` in the header.") };
    }

    pos += 9;

    bool littleEndian = (header[pos] == '<' || header[pos] == '|' ? true : false);
    if (!littleEndian) {
        throw invalid_argument{ tfm::format("Numpy image only supports little endian.") };
    }

    char type = header[pos + 1];
    int size = atoi(header.substr(pos + 2, 1).c_str()); // assume size <= 8 bytes
    if (type != 'f' && type != 'u' && size > 4) {
        throw invalid_argument{ tfm::format("Numpy image load error, type is neither `f` or `u`") };
    }

    // Order
    pos = header.find("fortran_order");
    if (pos == std::string::npos)
        throw invalid_argument{ tfm::format("Numpy image load error, no order information") };
    pos += 16;
    bool fortranOrder = header.substr(pos, 4) == "True" ? true : false;

    if (fortranOrder) // Only supports C order.
        throw invalid_argument{ tfm::format("Numpy image load error, only C order is supported now") };


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
        throw invalid_argument{ tfm::format("Numpy image load error, only supports numpy with shape length 2/3/4") };
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
        h = shape[1];
        w = shape[2];
        ch = shape[3];

    }
    if (ch > 4) // at most 4 channel
        throw invalid_argument{ tfm::format("Numpy image load error, only at most 4 channels is supported") };
    img_size = Vector2i(w, h);

    // load data
    std::vector<char> data;
    data.resize(w * h * ch * size);
    if (!iStream.read(data.data(), w * h * ch * size))
        throw invalid_argument{ tfm::format("Numpy image load error, cannot read the data region") };

    // convert raw data into float array

    std::vector<float> float_data;
    float_data.resize(w * h * ch);

    // type='f' size =2 ==> float16
    // type='f' size =4 ==> float32
    // type='u' size = 1 ==> uint8
    if (type == 'f' && size == 4) {
        const float * new_data = reinterpret_cast<const float*>(data.data());
        for (size_t i = 0; i < float_data.size(); ++i) {
            float_data[i] = new_data[i];
        }
    }
    else if (type == 'f' && size == 2) {
        const ::half* new_data = reinterpret_cast<const ::half*>(data.data());
        for (size_t i = 0; i < float_data.size(); ++i) {
            float_data[i] = float(new_data[i]);
        }
    }
    else if (type == 'u' && size == 1) {
        for (size_t i = 0; i < float_data.size(); ++i) {
            float_data[i] = float((unsigned char)(data[i])) / 255.0f;
        }
    }

    vector<Channel> channels = makeNChannels(ch, img_size);

    threadPool.parallelFor<DenseIndex>(0, img_size.y(), [&](DenseIndex y) {
        for (int x = 0; x < img_size.x(); ++x) {
            int baseIdx = (y * img_size.x() + x) * ch;
            for (int c = 0; c < ch; ++c) {
                float val = float_data[baseIdx + c];
                // Flip image vertically due to PFM format
                channels[c].at({ x, y }) = val;
            }
        }
    });

    vector<pair<size_t, size_t>> matches;
    for (size_t i = 0; i < channels.size(); ++i) {
        size_t matchId;
        if (matchesFuzzy(channels[i].name(), channelSelector, &matchId)) {
            matches.emplace_back(matchId, i);
        }
    }

    if (!channelSelector.empty()) {
        sort(begin(matches), end(matches));
    }

    for (const auto& match : matches) {
        result.channels.emplace_back(move(channels[match.second]));
    }

    // NPY can not contain layers, so all channels simply reside
    // within a topmost root layer.
    result.layers.emplace_back("");

    return result;
}

TEV_NAMESPACE_END
