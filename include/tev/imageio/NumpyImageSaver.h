// This file was developed by Duan Gao <gao-d17@mails.tsinghua.edu.cn>.
// It is published under the BSD 3-Clause License within the LICENSE file.

#pragma once

#include <tev/imageio/ImageSaver.h>

#include <ostream>

TEV_NAMESPACE_BEGIN

class NumpyImageSaver : public TypedImageSaver<float> {
public:
    void save(std::ostream& oStream, const filesystem::path& path, const std::vector<float>& data, const Eigen::Vector2i& imageSize, int nChannels) const override;

    bool hasPremultipliedAlpha() const override {
        return false;
    }

    virtual bool canSaveFile(const std::string& extension) const override {
        std::string lowerExtension = toLower(extension);
        return lowerExtension == "npy";
    }
};

TEV_NAMESPACE_END
