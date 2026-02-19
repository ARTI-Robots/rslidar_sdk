#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

struct DepthImageParams
{
  int width  = 0;   // output image width (pixels)
  int height = 0;   // output image height (pixels)

  float yaw_min_rad   = -static_cast<float>(M_PI) / 3.0f; // -60°
  float yaw_max_rad   = +static_cast<float>(M_PI) / 3.0f; // +60°
  float pitch_min_rad = -static_cast<float>(M_PI) / 4.0f; // -45°
  float pitch_max_rad = +static_cast<float>(M_PI) / 4.0f; // +45°


  float min_range = 0.1f;
  float max_range = 200.0f;
  void set(float w, float h, float min_r, float max_r, float yaw_min, float yaw_max, float pitch_min, float pitch_max)
  {
    width  = w;
    height = h;
    min_range = min_r;
    max_range = max_r;
    yaw_min_rad   = yaw_min * static_cast<float>(M_PI) / 180.0f;
    yaw_max_rad   = yaw_max * static_cast<float>(M_PI) / 180.0f;
    pitch_min_rad = pitch_min * static_cast<float>(M_PI) / 180.0f;
    pitch_max_rad = pitch_max * static_cast<float>(M_PI) / 180.0f;
  }
};

class DepthImageBuilder
{
public:
  explicit DepthImageBuilder(const DepthImageParams& p) : p_(p) {}

  const DepthImageParams& params() const { return p_; }

  template <class PointT>
  void buildAngular(const PointT* pts, std::size_t n_pts, float* depth_out) const
  {
    if (!pts || !depth_out) return;
    if (p_.width <= 0 || p_.height <= 0) return;

    const float yaw_span   = p_.yaw_max_rad   - p_.yaw_min_rad;
    const float pitch_span = p_.pitch_max_rad - p_.pitch_min_rad;
    if (!(yaw_span > 0.0f) || !(pitch_span > 0.0f)) return;

    const int W = p_.width;
    const int H = p_.height;

    // Precompute reciprocals for speed
    const float inv_yaw_span   = 1.0f / yaw_span;
    const float inv_pitch_span = 1.0f / pitch_span;

    for (std::size_t i = 0; i < n_pts; ++i)
    {
      const float x = static_cast<float>(pts[i].x);
      const float y = static_cast<float>(pts[i].y);
      const float z = static_cast<float>(pts[i].z);

      // Skip NaNs/Infs early
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

      // Direction angles
      const float xy = std::sqrt(x * x + y * y);
      const float yaw   = (xy > 0.0f) ? std::atan2(-y, x) : 0.0f;
      const float pitch = std::atan2(z, xy);


      if (yaw < p_.yaw_min_rad || yaw >= p_.yaw_max_rad) continue;
      if (pitch < p_.pitch_min_rad || pitch >= p_.pitch_max_rad) continue;

      float r = std::sqrt(x * x + y * y + z * z);
      if (!std::isfinite(r)) continue;
      if (r < p_.min_range || r > p_.max_range) continue;

      // Map angles to pixel indices
      // u increases with yaw; v decreases with pitch (top = max pitch)
      const float u_f = (yaw - p_.yaw_min_rad) * inv_yaw_span * static_cast<float>(W - 1);
      const float v_f = (p_.pitch_max_rad - pitch) * inv_pitch_span * static_cast<float>(H - 1);
      int u = static_cast<int>(std::round(u_f)); // nearest pixel
      int v = static_cast<int>(std::round(v_f));

      if (u < 0) u = 0; else if (u >= W) u = W - 1;
      if (v < 0) v = 0; else if (v >= H) v = H - 1;

      const std::size_t idx = static_cast<std::size_t>(v) * static_cast<std::size_t>(W)
                            + static_cast<std::size_t>(u);

      // Z-buffer: keep nearest
      const float prev = depth_out[idx];
      if (!std::isfinite(prev) || r < prev) depth_out[idx] = r;
    }
  }

  template <class PointCloudLike>
  void buildAngular(const PointCloudLike& points, float* depth_out) const
  {
    buildAngular(points.data(), points.size(), depth_out);
  }


  static inline bool validDepth(float d)
  {
    return std::isfinite(d) && d > 0.0f;
  }

  void fillHolesNearest(std::vector<float>& depth)
  {
  int radius = 1;
  int passes = radius;
  std::vector<float> tmp(depth.size());

  for (int pass = 0; pass < passes; ++pass)
  {
    tmp = depth;

    for (int y = 0; y < p_.height; ++y)
    {
      const int y0 = std::max(0, y - radius);
      const int y1 = std::min(p_.height - 1, y + radius);

      for (int x = 0; x < p_.width; ++x)
      {
        const size_t idx = (size_t)y * (size_t)p_.width + (size_t)x;
        if (validDepth(depth[idx])) continue;

        const int x0 = std::max(0, x - radius);
        const int x1 = std::min(p_.width - 1, x + radius);

        float best = std::numeric_limits<float>::infinity();
        int bestDist2 = 1e9;

        for (int yy = y0; yy <= y1; ++yy)
        {
          for (int xx = x0; xx <= x1; ++xx)
          {
            const size_t j = (size_t)yy * (size_t)p_.width + (size_t)xx;
            const float d = depth[j];
            if (!validDepth(d)) continue;

            const int dx = xx - x;
            const int dy = yy - y;
            const int dist2 = dx*dx + dy*dy;

            // Prefer nearest pixel; for ties prefer smaller depth (more conservative)
            if (dist2 < bestDist2 || (dist2 == bestDist2 && d < best))
            {
              bestDist2 = dist2;
              best = d;
            }
          }
        }

        if (std::isfinite(best))
          tmp[idx] = best;
      }
    }

    depth.swap(tmp);
    }
  }

private:
  DepthImageParams p_;
};
