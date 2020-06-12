

#include "mser_sift.h"




#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>





/************************************ MSER.h ********************************************/

#include <deque>
#include <vector>

#include <stdint.h>

/// The MSER class extracts maximally stable extremal regions from a grayscale (8 bit) image.
/// @note The MSER class is not reentrant, so if you want to extract regions in parallel, each
/// thread needs to have its own MSER class instance.
class MSER
{
public:
    /// A Maximally Stable Extremal Region.
    struct Region
    {
        int level;          ///< Level at which the region is processed.
        int pixel;          ///< Index of the initial pixel (y * width + x).
        int area;           ///< Area of the region (moment zero).
        double moments[5];  ///< First and second moments of the region (x, y, x^2, xy, y^2).
        double variation;   ///< MSER variation.

        /// Constructor.
        /// @param[in]  level  Level at which the region is being processed.
        /// @param[in]  pixel  Index of the initial pixel (y * width + x).
        Region(int level = 256,
               int pixel = -1);

        // Implementation details (could be moved outside of this header file)
    private:
        bool stable_;      // Flag indicating if the region is stable
        Region * parent_;  // Pointer to the parent region
        Region * child_;   // Pointer to the first child
        Region * next_;    // Pointer to the next (sister) region

        void accumulate(double x, double y);
        void merge(Region * child);
        void process(int delta, int minArea, int maxArea, double maxVariation, double minDiversity);
        void save(std::vector<Region> & regions) const;

        friend class MSER;
    };

    /// Constructor.
    /// @param[in]  delta         DELTA parameter of the MSER algorithm. Roughly speaking, the
    ///                           stability of a region is the relative variation of the region
    ///                           area when the intensity is changed by delta.
    /// @param[in]  minArea       Minimum area of any stable region in pixels.
    /// @param[in]  maxArea       Maximum area of any stable region in pixels.
    /// @param[in]  maxVariation  Maximum variation (absolute stability score) of the regions.
    /// @param[in]  minDiversity  Minimum diversity of the regions. When the relative area of two
    ///                           nested regions is above this threshold, then only the most stable
    ///                           one is selected.
    /// @param[in]  eight         Use 8-connected pixels instead of 4-connected.
    MSER(int delta = 1,
         int minArea = 100,
         int maxArea = 10000,
         double maxVariation = 0.75,
         double minDiversity = 0.5,
         bool eight = false);

    /// Extracts maximally stable extremal regions from a grayscale (8 bit) image.
    /// @param[in]  bits    Pointer to the first scanline of the image.
    /// @param[in]  width   Width of the image.
    /// @param[in]  height  Height of the image.
    /// @return  Detected MSERs.
    std::vector<Region> operator()(const uint8_t * bits,
                                   int width,
                                   int height) const;

    // Implementation details (could be moved outside of this header file)
private:
    // Helper method
    void processStack(int newPixelGreyLevel, int pixel, std::vector<Region *> & regionStack) const;

    // Parameters
    int delta_;
    int minArea_;
    int maxArea_;
    double maxVariation_;
    double minDiversity_;
    bool eight_;

    // Memory pool of regions for faster allocation
    mutable std::deque<Region> pool_;
    mutable std::size_t poolIndex_;
};

/************************************ MSER.h ********************************************/










/************************************ MSER.cpp ********************************************/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

using namespace std;

MSER::Region::Region(int level, int pixel)
: level(level), pixel(pixel), area(0), variation(numeric_limits<double>::infinity()),
  stable_(false), parent_(0), child_(0), next_(0)
{
    fill_n(moments, 5, 0.0);
}

void MSER::Region::accumulate(double x, double y)
{
    ++area;
    moments[0] += x;
    moments[1] += y;
    moments[2] += x * x;
    moments[3] += x * y;
    moments[4] += y * y;
}

void MSER::Region::merge(Region * child)
{
    assert(!child->parent_);
    assert(!child->next_);

    // Add the moments together
    area += child->area;
    moments[0] += child->moments[0];
    moments[1] += child->moments[1];
    moments[2] += child->moments[2];
    moments[3] += child->moments[3];
    moments[4] += child->moments[4];

    child->next_ = child_;
    child_ = child;
    child->parent_ = this;
}

void MSER::Region::process(int delta, int minArea, int maxArea, double maxVariation,
                           double minDiversity)
{
    // Find the last parent with level not higher than level + delta
    Region * parent = this;

    while (parent->parent_ && (parent->parent_->level <= level + delta))
        parent = parent->parent_;

    variation = static_cast<double>(parent->area - area) / area;
    stable_ = (area >= minArea) && (area <= maxArea) && (variation <= maxVariation);

    // Make sure the regions are diverse enough
    for (Region * p = parent_; p && (area > minDiversity * p->area); p = p->parent_) {
        if (p->variation <= variation)
            stable_ = false;

        if (variation < p->variation)
            p->stable_ = false;
    }

    // Process all the children
    for (Region * child = child_; child; child = child->next_)
        child->process(delta, minArea, maxArea, maxVariation, minDiversity);
}

void MSER::Region::save(vector<Region> & regions) const
{
    if (stable_)
        regions.push_back(*this);

    for (const Region * child = child_; child; child = child->next_)
        child->save(regions);
}

MSER::MSER(int delta, int minArea, int maxArea, double maxVariation, double minDiversity,
           bool eight)
: delta_(delta), minArea_(minArea), maxArea_(maxArea), maxVariation_(maxVariation),
  minDiversity_(minDiversity), eight_(eight), pool_(256), poolIndex_(0)
{
    // Parameter check
    assert(delta > 0);
    assert(minArea > 0);
    assert(maxArea >= minArea);
    assert(maxVariation >= 0.0);
    assert(minDiversity >= 0.0);
}

vector<MSER::Region> MSER::operator()(const uint8_t * bits, int width, int height) const
{
    if (!bits || (width <= 0) || (height <= 0))
        return vector<Region>();

    // 1. Clear the accessible pixel mask, the heap of boundary pixels and the component stack. Push
    // a dummy-component onto the stack, with grey-level higher than any allowed in the image.
    vector<bool> accessible(width * height);
    vector<int> boundaryPixels[256];
    int priority = 256;
    vector<Region *> regionStack;

    regionStack.push_back(new (&pool_[poolIndex_++]) Region);

    // 2. Make the source pixel (with its first edge) the current pixel, mark it as accessible and
    // store the grey-level of it in the variable current level.
    int curPixel = 0;
    int curEdge = 0;
    int curLevel = bits[0];

    accessible[0] = true;

    // 3. Push an empty component with current level onto the component stack.
step_3:
    regionStack.push_back(new (&pool_[poolIndex_++]) Region(curLevel, curPixel));

    if (poolIndex_ == pool_.size())
        pool_.resize(pool_.size() + 256);

    // 4. Explore the remaining edges to the neighbors of the current pixel, in order, as follows:
    // For each neighbor, check if the neighbor is already accessible. If it is not, mark it as
    // accessible and retrieve its grey-level. If the grey-level is not lower than the current one,
    // push it onto the heap of boundary pixels. If on the other hand the grey-level is lower than
    // the current one, enter the current pixel back into the queue of boundary pixels for later
    // processing (with the next edge number), consider the new pixel and its grey-level and go to
    // 3.
    for (;;) {
        const int x = curPixel % width;
        const int y = curPixel / width;

        const int offsets[8][2] = {
            { 1, 0}, { 0, 1}, {-1, 0}, { 0,-1},
            { 1, 1}, {-1, 1}, {-1,-1}, { 1,-1}
        };

        for (; curEdge < (eight_ ? 8 : 4); ++curEdge) {
            const int nx = x + offsets[curEdge][0];
            const int ny = y + offsets[curEdge][1];

            if ((nx >= 0) && (ny >= 0) && (nx < width) && (ny < height)) {
                const int neighborPixel = ny * width + nx;

                if (!accessible[neighborPixel]) {
                    const int neighborLevel = bits[neighborPixel];

                    accessible[neighborPixel] = true;

                    if (neighborLevel >= curLevel) {
                        boundaryPixels[neighborLevel].push_back(neighborPixel << 4);

                        if (neighborLevel < priority)
                            priority = neighborLevel;
                    }
                    else {
                        boundaryPixels[curLevel].push_back((curPixel << 4) | (curEdge + 1));

                        if (curLevel < priority)
                            priority = curLevel;

                        curPixel = neighborPixel;
                        curEdge = 0;
                        curLevel = neighborLevel;

                        goto step_3;
                    }
                }
            }
        }

        // 5. Accumulate the current pixel to the component at the top of the stack (water
        // saturates the current pixel).
        regionStack.back()->accumulate(x, y);

        // 6. Pop the heap of boundary pixels. If the heap is empty, we are done. If the returned
        // pixel is at the same grey-level as the previous, go to 4.
        if (priority == 256) {
            regionStack.back()->process(delta_, minArea_, maxArea_, maxVariation_, minDiversity_);

            vector<Region> regions;

            regionStack.back()->save(regions);

            poolIndex_ = 0;

            return regions;
        }

        curPixel = boundaryPixels[priority].back() >> 4;
        curEdge = boundaryPixels[priority].back() & 0xf;

        boundaryPixels[priority].pop_back();

        while (boundaryPixels[priority].empty() && (priority < 256))
            ++priority;

        // 7. The returned pixel is at a higher grey-level, so we must now process all components on
        // the component stack until we reach the higher grey-level. This is done with the
        // processStack sub-routine, see below.
        // Then go to 4.
        const int newPixelGreyLevel = bits[curPixel];

        if (newPixelGreyLevel != curLevel) {
            curLevel = newPixelGreyLevel;

            processStack(newPixelGreyLevel, curPixel, regionStack);
        }
    }
}

void MSER::processStack(int newPixelGreyLevel, int pixel, vector<Region *> & regionStack) const
{
    // 1. Process component on the top of the stack. The next grey-level is the minimum of
    // newPixelGreyLevel and the grey-level for the second component on the stack.
    do {
        Region * top = regionStack.back();

        regionStack.pop_back();

        // 2. If newPixelGreyLevel is smaller than the grey-level on the second component on the
        // stack, set the top of stack grey-level to newPixelGreyLevel and return from sub-routine
        // (This occurs when the new pixel is at a grey-level for which there is not yet a component
        // instantiated, so we let the top of stack be that level by just changing its grey-level.
        if (newPixelGreyLevel < regionStack.back()->level) {
            regionStack.push_back(new (&pool_[poolIndex_++]) Region(newPixelGreyLevel, pixel));

            if (poolIndex_ == pool_.size())
                pool_.resize(pool_.size() + 256);

            regionStack.back()->merge(top);

            return;
        }

        // 3. Remove the top of stack and merge it into the second component on stack as follows:
        // Add the first and second moment accumulators together and/or join the pixel lists.
        // Either merge the histories of the components, or take the history from the winner. Note
        // here that the top of stack should be considered one ’time-step’ back, so its current
        // size is part of the history. Therefore the top of stack would be the winner if its
        // current size is larger than the previous size of second on stack.
        regionStack.back()->merge(top);
    }
    // 4. If(newPixelGreyLevel>top of stack grey-level) go to 1.
    while (newPixelGreyLevel > regionStack.back()->level);
}

/************************************ MSER.cpp ********************************************/









/************************************ mipmap.h ********************************************/

#include <vector>

#include <stdint.h>

/// The Mipmap class stores a collection of reduced version of the input grayscale (8 bits) image,
/// and can be used to perform nearest neighbor, bilinear and trilinear sampling (with clamp to
/// edge).
class Mipmap
{
public:
    /// Constructor.
    /// @param[in]  bits    Pointer to the first scanline of the image.
    /// @param[in]  width   Width of the image.
    /// @param[in]  height  Height of the image.
    /// @note  The @p bits pointer must remain valid for the entire lifetime of the @c Mipmap.
    Mipmap(const uint8_t * bits,
           int width,
           int height);

    /// Returns the number of levels.
    int numberOfLevels() const;

    /// Returns a pointer to the first scanline of level @p l.
    const uint8_t * level(int l) const;

    /// Nearest neighbor sampling.
    /// @param[in]  x  Abscissa.
    /// @param[in]  y  Ordinate.
    /// @param[in]  l  Level (scale).
    uint8_t operator()(int x,
                       int y,
                       int l = 0) const;

    /// Bilinear sampling.
    /// @param[in]  x  Abscissa.
    /// @param[in]  y  Ordinate.
    /// @param[in]  l  Level (scale).
    double operator()(double x,
                      double y,
                      int l = 0) const;

    /// Trilinear sampling.
    /// @param[in]  x  Abscissa.
    /// @param[in]  y  Ordinate.
    /// @param[in]  l  Level (scale).
    double operator()(double x,
                      double y,
                      double l) const;

private:
    std::vector<const uint8_t *> levels_;
    std::vector<uint8_t> data_;
    int width_;
    int height_;
    int numberOfLevels_;
};

/************************************ mipmap.h ********************************************/








/************************************ mipmap.cpp ********************************************/

#include <algorithm>
#include <cassert>
#include <cmath>

using namespace std;

Mipmap::Mipmap(const uint8_t * bits, int width, int height)
: width_(width), height_(height), numberOfLevels_(0)
{
    if (!bits || (width <= 0) || (height <= 0))
        return;

    // Compute the number of levels and the memory required to store them
    size_t size = 0;

    while (width && height) {
        ++numberOfLevels_;
        width >>= 1;
        height >>= 1;
        size += width * height;
    }

    // The first level is the original image
    levels_.push_back(bits);

    // The additional storage size required
    data_.resize(size);

    // Fill every level
    width = width_;
    height = height_;
    const uint8_t * src = bits;
    uint8_t * dst = &data_[0];
    vector<uint8_t> tmp((width >> 1) * height);

    for (int l = 1; l < numberOfLevels_; ++l) {
        // Index the level in the data
        levels_.push_back(dst);

        // Blur src's rows into tmp (sigma = 1)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < (width >> 1); ++x) {
                const int x0 = min(max(x * 2 - 2, 0), width - 1);
                const int x1 = min(max(x * 2 - 1, 0), width - 1);
                const int x2 = min(max(x * 2    , 0), width - 1);
                const int x3 = min(max(x * 2 + 1, 0), width - 1);
                const int x4 = min(max(x * 2 + 2, 0), width - 1);
                const int x5 = min(max(x * 2 + 3, 0), width - 1);
                const int a = src[y * width + x0] + src[y * width + x5];
                const int b = src[y * width + x1] + src[y * width + x4];
                const int c = src[y * width + x2] + src[y * width + x3];

                tmp[x * height + y] = (1151 * a + 8503 * b + 23114 * c + 32768) >> 16;
            }
        }

        width >>= 1;

        // Blur tmp's columns into dst (sigma = 1)
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < (height >> 1); ++y) {
                const int y0 = min(max(y * 2 - 2, 0), height - 1);
                const int y1 = min(max(y * 2 - 1, 0), height - 1);
                const int y2 = min(max(y * 2    , 0), height - 1);
                const int y3 = min(max(y * 2 + 1, 0), height - 1);
                const int y4 = min(max(y * 2 + 2, 0), height - 1);
                const int y5 = min(max(y * 2 + 3, 0), height - 1);
                const int a = tmp[x * height + y0] + tmp[x * height + y5];
                const int b = tmp[x * height + y1] + tmp[x * height + y4];
                const int c = tmp[x * height + y2] + tmp[x * height + y3];

                dst[y * width + x] = (1151 * a + 8503 * b + 23114 * c + 32768) >> 16;
            }
        }

        height >>= 1;

        // The source of the next level is the current level
        src = levels_.back();
        dst += width * height;
    }
}

int Mipmap::numberOfLevels() const
{
    return numberOfLevels_;
}

const uint8_t * Mipmap::level(int l) const
{
    // Clamp the level
    l = min(max(l, 0), numberOfLevels_ - 1);

    return levels_[l];
}

uint8_t Mipmap::operator()(int x, int y, int l) const
{
    // Clamp the level
    l = min(max(l, 0), numberOfLevels_ - 1);

    // Clamp the coordinates
    x = min(max(x, 0), width_ - 1);
    y = min(max(y, 0), height_ - 1);

    // Convert to level l
    return levels_[l][(y >> l) * (width_ >> l) + (x >> l)];
}

double Mipmap::operator()(double x, double y, int l) const
{
    // Clamp the level
    l = min(max(l, 0), numberOfLevels_ - 1);

    x = ldexp(x, -l);
    y = ldexp(y, -l);

    // Clamp the coordinates
    const int width = width_ >> l;
    const int height = height_ >> l;

    x = min(max(x, 0.0), width - 1.0);
    y = min(max(y, 0.0), height - 1.0);

    // Bilinear interpolation
    const int x0 = x;
    const int x1 = min(x0 + 1, width - 1);
    const int y0 = y;
    const int y1 = min(y0 + 1, height - 1);
    const double a = x - x0;
    const double b = 1.0 - a;
    const double c = y - y0;
    const double d = 1.0 - c;

    return (levels_[l][y0 * width + x0] * b + levels_[l][y0 * width + x1] * a) * d +
           (levels_[l][y1 * width + x0] * b + levels_[l][y1 * width + x1] * a) * c;
}

double Mipmap::operator()(double x, double y, double l) const
{
    // Clamp the level
    if (l <= 0.0)
        return operator()(x, y, 0);
    else if (l >= numberOfLevels_ - 1.0)
        return operator()(x, y, numberOfLevels_ - 1);

    // Interpolation of the two closest levels
    const int l0 = l;
    const int l1 = l0 + 1;
    const double a = l - l0;
    const double b = 1.0 - a;

    return operator()(x, y, l0) * b + operator()(x, y, l1) * a;
}

/************************************ mipmap.cpp ********************************************/










/************************************ affine.h ********************************************/

#include <vector>

#include <stdint.h>

/// The Affine class extracts multiple affine regions from a grayscale (8 bit) image.
class Affine
{
public:
    /// An affine region.
    struct Region
    {
        double x;      ///< Center.
        double y;      ///< Center.
        double a;      ///< Covariance matrix [a b; b c].
        double b;      ///< Covariance matrix [a b; b c].
        double c;      ///< Covariance matrix [a b; b c].
        double angle;  ///< Angle of the x-axis relative to the principal axis.
    };

    /// Constructor.
    /// @param[in]  resolution  Resolution (in pixels) at which the regions shoud be extracted.
    ///                         Must be a multiple of 4 in the range [4, 256].
    /// @param[in]  radius      Number of standard deviations of the ellipses fit to the regions.
    Affine(int resolution = 64,
           double radius = 3.0);

    /// @brief Extracts multiple affine regions from a grayscale (8 bit) image.
    /// @param[in]  bits     Pointer to the first scanline of the image.
    /// @param[in]  width    Width of the image.
    /// @param[in]  height   Height of the image.
    /// @param[in]  regions  Affine regions to extract from the image.
    /// @return  Image of the concatenated affine regions of dimensions
    ///          resolution x (regions.size() * resolution).
    std::vector<uint8_t> operator()(const uint8_t * bits,
                                    int width,
                                    int height,
                                    const std::vector<Region> & regions);

    // Implementation details (could be moved outside of this header file)
private:
    // Parameters
    const int resolution_;
    const double radius_;
};

/************************************ affine.h ********************************************/









/************************************ affine.cpp ********************************************/

#include <algorithm>
#include <cassert>
#include <cmath>

using namespace std;

Affine::Affine(int resolution, double radius)
: resolution_(resolution), radius_(radius)
{
    assert(resolution > 0);
    assert(radius > 0.0);
}

vector<uint8_t> Affine::operator()(const uint8_t * bits, int width, int height,
                                   const vector<Region> & regions)
{
    if (!bits || (width <= 0) || (height <= 0) || regions.empty())
        return vector<uint8_t>();

    const Mipmap mipmap(bits, width, height);

    // Create the output image
    vector<uint8_t> image(regions.size() * resolution_ * resolution_);

    for (size_t i = 0; i < regions.size(); ++i) {
        // Square root of the covariance matrix
        const double trace = regions[i].a + regions[i].c;
        const double sqrtDet = sqrt(regions[i].a * regions[i].c - regions[i].b * regions[i].b);
        const double alpha = 2.0 * radius_ / resolution_ / sqrt(trace + 2.0 * sqrtDet);
        const double c = alpha * cos(regions[i].angle);
        const double s = alpha * sin(regions[i].angle);

        double affine[2][2] = {
            { c * (regions[i].a + sqrtDet) + s * regions[i].b,
             -s * (regions[i].a + sqrtDet) + c * regions[i].b},
            { c * regions[i].b + s * (regions[i].c + sqrtDet),
             -s * regions[i].b + c * (regions[i].c + sqrtDet)}
        };

        const double scale = 0.5 * log2(affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0]);

        // Recopy the buffer into the output image
        for (int v = 0; v < resolution_; ++v) {
            for (int u = 0; u < resolution_; ++u) {
                const double u2 = u - 0.5 * (resolution_ - 1);
                const double v2 = v - 0.5 * (resolution_ - 1);

                image[(i * resolution_ + v) * resolution_ + u] =
                        mipmap(affine[0][0] * u2 + affine[0][1] * v2 + regions[i].x,
                               affine[1][0] * u2 + affine[1][1] * v2 + regions[i].y, scale);
            }
        }
    }

    return image;
}

/************************************ affine.cpp ********************************************/













/************************************ SIFT.h ********************************************/

/// The SIFT class computes the SIFT (Scale-Invariant Feature Transform) descriptor of the regions
/// extracted from a grayscale (8 bit) image.
class SIFT
{
public:
    /// A SIFT descriptor.
    struct Descriptor
    {
        double x;                   ///< Center.
        double y;                   ///< Center.
        double a;                   ///< Covariance matrix [a b; b c].
        double b;                   ///< Covariance matrix [a b; b c].
        double c;                   ///< Covariance matrix [a b; b c].
        double angle;               ///< Angle of the x-axis relative to the principal axis.
        std::vector<uint8_t> data;  ///< Descriptor data.
    };

    /// Constructor.
    /// @param[in]  resolution  Resolution (in pixels) at which the regions shoud be extracted.
    ///                         Must be a multiple of 4 in the range [4, 256].
    /// @param[in]  radius      Number of standard deviations of the ellipses fit to the regions.
    SIFT(int resolution = 84,
         double radius = 3.0);

    /// Computes the SIFT descriptor of the regions extracted from a grayscale (8 bit) image.
    /// @param[in]  bits                  Pointer to the first scanline of the image.
    /// @param[in]  width                 Width of the image.
    /// @param[in]  height                Height of the image.
    /// @param[in]  regions               Detected MSERs.
    /// @param[in]  orientationInvariant  Whether to compute a descriptor for each dominant
    ///                                   orientation or only for the original one.
    /// @return  The descriptor associated to each region.
    std::vector<Descriptor> operator()(const uint8_t * bits,
                                       int width,
                                       int height,
                                       const std::vector<MSER::Region> & regions,
                                       bool orientationInvariant = true) const;

    // Implementation details (could be moved outside this header file)
private:
    // Parameters
    int resolution_;
    double radius_;

    // Lookup tables
    double sqrtTable_[512][512];
    double atan2Table_[512][512];
    std::vector<double> siftTables_[4][4];
    std::vector<int> minMaxTables_[4];
};
/************************************ SIFT.h ********************************************/








/************************************ SIFT.cpp ********************************************/

#include <algorithm>
#include <cmath>


using namespace std;

SIFT::SIFT(int resolution,
           double radius)
: resolution_(resolution), radius_(radius)
{
    // Fill the sqrt and atan2 tables
    for (int i = -255; i <= 255; ++i) {
        for (int j = -255; j <= 255; ++j) {
            sqrtTable_[i + 255][j + 255] = sqrt(i * i + j * j);

            double a = 0.5 * (atan2(i, j) / M_PI + 1.0);

            if (a == 1.0)
                a = 0.0;

            atan2Table_[i + 255][j + 255] = a;
        }
    }

    // Fill the SIFT interpolation tables
    minMaxTables_[0].resize(resolution * resolution, 3);
    minMaxTables_[1].resize(resolution * resolution, 0);
    minMaxTables_[2].resize(resolution * resolution, 3);
    minMaxTables_[3].resize(resolution * resolution, 0);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            siftTables_[i][j].resize(resolution * resolution);

            for (int v = 0; v < resolution; ++v) {
                for (int u = 0; u < resolution; ++u) {
                    const int k = v * resolution + u;

                    siftTables_[i][j][k] =
                            max(1.0 - abs(4.0 * (u + 0.5) / resolution - j - 0.5), 0.0) *
                            max(1.0 - abs(4.0 * (v + 0.5) / resolution - i - 0.5), 0.0);

                    if (siftTables_[i][j][k] > 0.0) {
                        minMaxTables_[0][k] = min(minMaxTables_[0][k], j);
                        minMaxTables_[1][k] = max(minMaxTables_[1][k], j);
                        minMaxTables_[2][k] = min(minMaxTables_[2][k], i);
                        minMaxTables_[3][k] = max(minMaxTables_[3][k], i);
                    }
                }
            }
        }
    }
}

// Normalize a descriptor by dividing it by its norm, clamping it to 0.2, dividing it by its norm
// again, and finally clamping it to 0.3 before converting it to the range [0, 255]
static double normalize(double * x,
                        int n)
{
    double sumSquared = 0.0;

    for (int i = 0; i < n; ++i)
        sumSquared += x[i] * x[i];

    const double norm = sqrt(sumSquared);
    double invNorm = 1.0 / norm;

    sumSquared = 0.0;

    for (int i = 0; i < n; ++i) {
        x[i] = min(x[i] * invNorm, 0.2);
        sumSquared += x[i] * x[i];
    }

    invNorm = 1.0 / sqrt(sumSquared);

    for (int i = 0; i < n; ++i)
        x[i] = min((256.0 / 0.3) * x[i] * invNorm, 255.0);

    return norm;
}

vector<SIFT::Descriptor> SIFT::operator()(const uint8_t * bits,
                                          int width,
                                          int height,
                                          const std::vector<MSER::Region> & mserRegions,
                                          bool orientationInvariant) const
{
    if (!bits || (width <= 0) || (height <= 0) || mserRegions.empty())
        return vector<Descriptor>();

    // Convert the MSER regions to Affine regions
    vector<Affine::Region> regions;

    for (size_t i = 0; i < mserRegions.size(); ++i) {
        const double x = mserRegions[i].moments[0] / mserRegions[i].area;
        const double y = mserRegions[i].moments[1] / mserRegions[i].area;
        const double a = (mserRegions[i].moments[2] - x * mserRegions[i].moments[0]) /
                         (mserRegions[i].area - 1);
        const double b = (mserRegions[i].moments[3] - x * mserRegions[i].moments[1]) /
                         (mserRegions[i].area - 1);
        const double c = (mserRegions[i].moments[4] - y * mserRegions[i].moments[1]) /
                         (mserRegions[i].area - 1);

        // Skip too elongated regions
        const double d = sqrt((a - c) * (a - c) + 4.0 * b * b);

        if ((a + c + d) / (a + c - d) < 25.0) {
            const Affine::Region region = {x, y, a, b, c, 0.0};

            regions.push_back(region);
        }
    }

    // Extract the regions from the image
    Affine affine(resolution_, radius_);

    vector<uint8_t> image = affine(bits, width, height, regions);

    if (image.empty())
        return vector<Descriptor>();

    // Compute the dominant orientations if needed
    if (orientationInvariant) {
        for (size_t i = 0; i < regions.size(); ++i) {
            const uint8_t * pixels = &image[i * resolution_ * resolution_];

            // Compute the gradient of the bounding box
            double hist[36] = {};

            for (int v = 0; v < resolution_; ++v) {
                for (int u = 0; u < resolution_; ++u) {
                    const double r2 =
                            (u - resolution_ / 2.0 + 0.5) * (u - resolution_ / 2.0 + 0.5) +
                            (v - resolution_ / 2.0 + 0.5) * (v - resolution_ / 2.0 + 0.5);

                    if (r2 < resolution_ * resolution_ / 4.0) {
                        const int du = pixels[v * resolution_ + min(u + 1, resolution_ - 1)] -
                                       pixels[v * resolution_ + max(u - 1, 0)];
                        const int dv = pixels[min(v + 1, resolution_ - 1) * resolution_ + u] -
                                       pixels[max(v - 1, 0) * resolution_ + u];
                        const double n = sqrtTable_[dv + 255][du + 255];
                        const double a = 36.0 * atan2Table_[dv + 255][du + 255];
                        const int a0 = a;
                        const int a1 = (a0 + 1) % 36;
                        const double z = exp(-r2 / (resolution_ * resolution_ / 4.0));

                        hist[a0] += (1.0 - (a - a0)) * (n * z);
                        hist[a1] +=		   (a - a0)  * (n * z);
                    }
                }
            }

            // Blur the histogram (sigma = 1, applied 6 times)
            for (int j = 0; j < 3; ++j) {
                double tmp[36];

                for (int k = 0; k < 36; ++k)
                    tmp[k] = 0.054489 * (hist[(k + 34) % 36] + hist[(k + 2) % 36]) +
                             0.244201 * (hist[(k + 35) % 36] + hist[(k + 1) % 36]) +
                             0.402620 * hist[k];

                for (int k = 0; k < 36; ++k)
                    hist[k] = 0.054489 * (tmp[(k + 34) % 36] + tmp[(k + 2) % 36]) +
                              0.244201 * (tmp[(k + 35) % 36] + tmp[(k + 1) % 36]) +
                              0.402620 * tmp[k];
            }

            // Add a descriptor for each local maximum greater than 80% of the global maximum
            const double maxh = *max_element(hist, hist + 36);

            for (int j = 0; j < 36; ++j) {
                const double h0 = hist[j];
                const double hm = hist[(j + 35) % 36];
                const double hp = hist[(j +  1) % 36];

                if ((h0 > 0.8 * maxh) && (h0 > hm) && (h0 > hp)) {
                    regions[i].angle = (j - 0.5 * (hp - hm) / (hp + hm - 2.0 * h0)) * M_PI / 18.0;
                }
            }
        }

        image = affine(bits, width, height, regions);

        if (image.empty())
            return vector<Descriptor>();
    }

    vector<Descriptor> descriptors;

    for (size_t i = 0; i < regions.size(); ++i) {
        const uint8_t * pixels = &image[i * resolution_ * resolution_];

        // Compute the SIFT descriptor
        double desc[128] = {};

        for (int v = 0; v < resolution_; ++v) {
            for (int u = 0; u < resolution_; ++u) {
                const int du = pixels[v * resolution_ + min(u + 1, resolution_ - 1)] -
                               pixels[v * resolution_ + max(u - 1, 0)];
                const int dv = pixels[min(v + 1, resolution_ - 1) * resolution_ + u] -
                               pixels[max(v - 1, 0) * resolution_ + u];
                const double n = sqrtTable_[dv + 255][du + 255];
                const double a = 8.0 * atan2Table_[dv + 255][du + 255];
                const int a0 = a;
                const int a1 = (a0 + 1) % 8;
                const int l = v * resolution_ + u;

                for (int j = minMaxTables_[2][l]; j <= minMaxTables_[3][l]; ++j) {
                    for (int k = minMaxTables_[0][l]; k <= minMaxTables_[1][l]; ++k) {
                        desc[(j * 4 + k) * 8 + a0] += siftTables_[j][k][l] * (1.0 - (a - a0)) * n;
                        desc[(j * 4 + k) * 8 + a1] += siftTables_[j][k][l] *        (a - a0)  * n;
                    }
                }
            }
        }

        // Normalize the descriptor and skip regions of too low gradient
        if (normalize(desc, 128) < resolution_ * resolution_)
            continue;

        const Descriptor descriptor = {regions[i].x, regions[i].y, regions[i].a, regions[i].b,
                                       regions[i].c, regions[i].angle,
                                       vector<uint8_t>(desc, desc + 128)};

        descriptors.push_back(descriptor);
    }

    return descriptors;
}

/************************************ SIFT.cpp ********************************************/
























/************************************ ********** ********************************************/
/************************************ ********** ********************************************/

#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <jpeglib.h>


using namespace std;

/*static bool loadJpeg(const char * filename, int & width, int & height, int & depth,
                     vector<uint8_t> & bits)
{
    // Try to load the jpeg image
    FILE * file = fopen(filename, "rb");

    if (!file) {
        cerr << "Could not open file: " << filename << endl;
        return false;
    }

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);

    if ((jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) || (cinfo.data_precision != 8) ||
        !jpeg_start_decompress(&cinfo)) {
        cerr << "Could not decompress jpeg file: " << filename << endl;
        fclose(file);
        return false;
    }

    width = cinfo.image_width;
    height = cinfo.image_height;
    depth = cinfo.num_components;
    bits.resize(width * height * depth);

    for (int y = 0; y < height; ++y) {
        JSAMPLE * row = static_cast<JSAMPLE *>(&bits[y * width * depth]);

        if (jpeg_read_scanlines(&cinfo, &row, 1) != 1) {
            cerr << "Could not decompress jpeg file: " << filename << endl;
            fclose(file);
            return false;
        }
    }

    jpeg_finish_decompress(&cinfo);
    fclose(file);

    return true;
}

static bool saveJpeg(const char * filename, int width, int height, int depth,
                     const vector<uint8_t> & bits)
{
    FILE * file = fopen(filename, "wb");

    if (!file) {
        cerr << "Could not open file: " << filename << endl;
        return false;
    }

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, file);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = depth;
    cinfo.in_color_space = (depth == 1) ? JCS_GRAYSCALE : JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 100, FALSE);
    jpeg_start_compress(&cinfo, TRUE);

    for (int y = 0; y < height; ++y) {
        const JSAMPLE * row = static_cast<const JSAMPLE *>(&bits[y * width * depth]);
        jpeg_write_scanlines(&cinfo, const_cast<JSAMPARRAY>(&row), 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(file);

    return true;
}

void drawEllipse(const MSER::Region & region, int width, int height, int depth,
                 const uint8_t color[3], vector<uint8_t> & bits)
{
    // Centroid (mean)
    const double x = region.moments[0] / region.area;
    const double y = region.moments[1] / region.area;

    // Covariance matrix [a b; b c]
    const double a = region.moments[2] / region.area - x * x;
    const double b = region.moments[3] / region.area - x * y;
    const double c = region.moments[4] / region.area - y * y;

    // Square root of the covariance matrix
    const double tr = a + c;
    const double sqrtDet = sqrt(a * c - b * b);
    const double d = (a + sqrtDet) / sqrt(tr + 2.0 * sqrtDet);
    const double e = b / sqrt(tr + 2.0 * sqrtDet);
    const double f = (c + sqrtDet) / sqrt(tr + 2.0 * sqrtDet);

    for (double t = 0.0; t < 2.0 * M_PI; t += 0.001) {
        const int x2 = x + (cos(t) * d + sin(t) * e) * 2.0 + 0.5;
        const int y2 = y + (cos(t) * e + sin(t) * f) * 2.0 + 0.5;

        if ((x2 >= 0) && (x2 < width) && (y2 >= 0) && (y2 < height))
            for (int i = 0; i < std::min(depth, 3); ++i)
                bits[(y2 * width + x2) * depth + i] = color[i];
    }
}*/

int* load_hsi(char* path, int* width, int* height, int* bands)
{
    FILE *fp;
    int *datos;
    int H = 0, V = 0, B = 0;
    size_t a;


    fp = fopen(path, "rb");
    if (fp == NULL) {
        //print_error((char*)"Can not open the image");
        exit(EXIT_FAILURE);
    }

    a = fread(&B, sizeof(int), 1, fp); *bands=B;
    a = fread(&H, sizeof(int), 1, fp); *width=H;
    a = fread(&V, sizeof(int), 1, fp); *height=V;

    datos = (int *) malloc(H*V*B * sizeof (int));
    if (datos == NULL) {
      //print_error((char*)"Not enough memory\n");
      exit(EXIT_FAILURE);
    }

    a = fread(datos, sizeof(int), (size_t) B * H*V, fp);
    if (a != (size_t) B * H * V) {
      //print_error((char*)"Read failure\n");
      exit(EXIT_FAILURE);
    }

    fclose(fp);
    return (datos);
}






descriptor_model_t mser_sift_features ( image_struct * image, unsigned int * seg, float* parameters )
{

		// Try to load the jpeg image
		int width=get_image_width(image);
		int height=get_image_height(image);
		int bands=get_image_bands(image);
		//vector<uint8_t> original;
		int *original2;
		int min_value, max_value;
		int nsegs=0;
    descriptor_model_t output;

		//original1 = load_hsi((char*) "../../archivos_pesados/data/Salinas_multi.raw", &width, &height, &bands);

    for(unsigned int sg=0;sg < get_image_width(image)*get_image_height(image);sg++){
      if(seg[sg] > (unsigned int)nsegs){
        nsegs = seg[sg];
      }
    }

    output.num_segments = nsegs+1;
    output.descriptors = new std::vector<Ds>[output.num_segments];
    output.descriptors_per_segment = new int[output.num_segments];


		//rescale: rescales the values in the range [0.0, 255.0] as the algorithm is tuned for PGM images
    find_maxmin(get_image_data(image),   height*width*bands,  (long long unsigned int*)&min_value,  (long long unsigned int*)&max_value);
    original2 = (int *) malloc(width*height*bands * sizeof (int)) ;
    for( int j=0; j<width*height*bands;j++){
      original2[j] = (int) 0 + ( (int)(((int)get_image_data(image)[j] - min_value)*(255-0)) / (int)(max_value-min_value) );
    }

		vector<uint8_t> grayscale_aux(width * height * bands);

		for (int i = 0; i < width * height * bands; ++i) {
			grayscale_aux[i] = (uint8_t) original2[ i ];
		}


    for(unsigned int band=0; band<get_image_bands(image); band++){

      printf("\n\n\t** Band %d **\n", band);

			vector<uint8_t> grayscale(width * height);

			for (int i = 0; i < width * height; ++i) {
				grayscale[i] = grayscale_aux[ (band*width*height) + i ];
			}

			// Extract MSER
			MSER mser(2, 50, width * height / 10, 0.1, 0.5);

			vector<MSER::Region> regions[2];

			clock_t start = clock();

			regions[0] = mser(&grayscale[0], width, height);

			// Invert the pixel values
			for (int i = 0; i < width * height; ++i)
					grayscale[i] = ~grayscale[i];

			regions[1] = mser(&grayscale[0], width, height);

			clock_t stop = clock();

			cout << "Extracted " << (regions[0].size() + regions[1].size()) << " regions from " << "XXX"
					 << " (" << width << 'x' << height << ") in "
					 << (static_cast<double>(stop - start) / CLOCKS_PER_SEC) << "s." << endl;

			for (int i = 0; i < width * height; ++i)
			   grayscale[i] = ~grayscale[i];

			SIFT sift;

			vector<SIFT::Descriptor> descriptors[2];

			start = clock();

			descriptors[0] = sift(&grayscale[0], width, height, regions[0], false);
			descriptors[1] = sift(&grayscale[0], width, height, regions[1], false);

			stop = clock();

			cout << "Described " << (descriptors[0].size() + descriptors[1].size()) << " regions from "
			    << "XXX" << " (" << width << 'x' << height << ") in "
			    << (static_cast<double>(stop - start) / CLOCKS_PER_SEC) << "s." << endl;

					for (int i = 0; i < 2; ++i) {
			        for (size_t j = 0; j < descriptors[i].size(); ++j) {
			            //cout << "\n\n" << descriptors[i][j].x << ' '
			            //    << descriptors[i][j].y << '\n';

									Ds descriptor;

			            for (int k = 0; k < 128; ++k)
											descriptor.desc.push_back((double) descriptors[i][j].data[k]);
			                //cout << ' ' << static_cast<int>(descriptors[i][j].data[k]);

			            //cout << endl;
									output.descriptors[seg[(int)descriptors[i][j].y * get_image_width(image) + (int)descriptors[i][j].x]].push_back(descriptor);
			        }
			    }

      printf("\n");
    }


    output.total_descriptors = 0;
    for(int ns=0;ns<output.num_segments;ns++){
      output.descriptors_per_segment[ns] = (int)output.descriptors[ns].size();
      output.total_descriptors += (int)output.descriptors[ns].size();
    }

		//printf("** %d\n", output.total_descriptors);

    return  output  ;
}
