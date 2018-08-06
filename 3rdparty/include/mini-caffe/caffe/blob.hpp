#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/base.hpp"

const int kMaxBlobAxes = 32;

namespace caffe {

class BlobShape;
class BlobProto;
class SyncedMemory;

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
class CAFFE_API Blob {
 public:
  Blob()
      : data_(), count_(0), capacity_(0), name_("") {}

  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels,
                const int height, const int width);
  explicit Blob(const vector<int>& shape);

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels,
               const int height, const int width);
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  void Reshape(const vector<int>& shape);
  void Reshape(const BlobShape& shape);
  void ReshapeLike(const Blob& other);
  std::string shape_string() const {
    std::ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }
  const vector<int>& shape() const { return shape_; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }
  int num_axes() const { return static_cast<int>(shape_.size()); }
  int count() const { return count_; }
  int capacity() const { return capacity_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  int width() const { return LegacyShape(3); }
  int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }

  int offset(const int n, const int c = 0,
             const int h = 0, const int w = 0) const {
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob& source, bool reshape = false);

  real_t data_at(const int n, const int c=0,
                 const int h=0, const int w=0) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  real_t data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  const int* gpu_shape() const;
  const real_t* cpu_data() const;
  const real_t* gpu_data() const;
  real_t* mutable_cpu_data();
  real_t* mutable_gpu_data();

  void FromProto(const BlobProto& proto, bool reshape = true);
  void ToProto(BlobProto* proto) const;

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);

  bool ShapeEquals(const BlobProto& other);

  std::string name() { return name_; }
  void set_name(std::string name) { name_ = name; }

 protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> shape_data_;
  vector<int> shape_;
  int count_;
  int capacity_;
  std::string name_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

/// @brief Read binary data
CAFFE_API shared_ptr<Blob> ReadBlobFromFile(const string& file);
CAFFE_API shared_ptr<Blob> ReadBlobFromBuffer(const string& buffer);

class BlobInt : public Blob {
 public:
  BlobInt()
    : Blob() {}

  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit BlobInt(const int num, const int channels,
                   const int height, const int width)
    : Blob(num, channels, height, width) {}
  explicit BlobInt(const vector<int>& shape)
    : Blob(shape) {}

  int data_at(const int n, const int c=0,
              const int h=0, const int w=0) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  int data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  const int* cpu_data() const;
  const int* gpu_data() const;
  int* mutable_cpu_data();
  int* mutable_gpu_data();

  void FromProto(const BlobProto& proto, bool reshape = true) = delete;
  void ToProto(BlobProto* proto) const = delete;

 protected:
  DISABLE_COPY_AND_ASSIGN(BlobInt);
};

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
