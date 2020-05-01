import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description
from onnx_tf.common.tf_helper import tf_shape


@onnx_op("Resize")
@partial_support(True)
@ps_description("Resize required 4D input in Tensorflow.")
class Resize(BackendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = x.get_shape().as_list()
    if len(x_shape) != 4:
      exception.OP_UNSUPPORTED_EXCEPT("Resize required 4D input", "Tensorflow")
    if cls.SINCE_VERSION >= 11:
      # TODO: need to add the content of the following table into the partial support description
      # supported attributes combination
      # ____________________________________________________________________________________________________________________________________________________
      # | mode    | coordinate_transformation_mode | cubic_coeff_a | exclude_outside | extrapolation_value | nearest_mode      | scales        | sizes     |
      # |_________|________________________________|_______________|_________________|_____________________|___________________|_______________|___________|
      # | nearest | tf_half_pixel_for_nn           | not apply     | 0               | not apply           | floor             | not supported | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | linear  | half_pixel                     | not apply     | 0               | not apply           | not apply         | not supported | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | cubic   | half_pixel                     | -0.5          | 1               | not apply           | not apply         | not supported | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | nearest | tf_crop_and_resize             | not apply     | 0               | any float value     | round_prefer_ceil | supported     | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | linear  | tf_crop_and_resize             | not apply     | 0               | any float value     | not apply         | supported     | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      coordinate_transformation_mode = node.attrs.get(
          "coordinate_transformation_mode", "half_pixel")
      cubic_coeff_a = node.attrs.get("cubic_coeff_a", -0.75)
      exclude_outside = node.attrs.get("exclude_outside", 0)
      extrapolation_value = node.attrs.get("extrapolation_value", 0.0)
      mode = node.attrs.get("mode", "nearest")
      nearest_mode = node.attrs.get("nearest_mode", "round_prefer_floor")
      if coordinate_transformation_mode in [
          "pytorch_half_pixel", "align_corners", "asymmetric"
      ]:
        exception.OP_UNSUPPORTED_EXCEPT(
            "Resize coordinate_transformation_mode=" +
            coordinate_transformation_mode, "Tensorflow")
      elif coordinate_transformation_mode == "tf_half_pixel_for_nn":
        if mode in ["linear", "cubic"]:
          exception.OP_UNSUPPORTED_EXCEPT(
              "Resize coordinate_transformation_mode=tf_half_pixel_for_nn " +
              "and mode=" + mode, "Tensorflow")
        if nearest_mode in ["round_prefer_floor", "round_prefer_ceil", "ceil"]:
          exception.OP_UNSUPPORTED_EXCEPT(
              "Resize coordinate_transformation_mode=tf_half_pixel_for_nn " +
              "and nearest_mode=" + nearest_mode, "Tensorflow")
        if len(node.inputs) == 3:  # sizes is not defined
          exception.OP_UNSUPPORTED_EXCEPT(
              "Resize coordinate_transformation_mode=tf_half_pixel_for_nn " +
              "with scales", "Tensorflow")
      elif coordinate_transformation_mode == "half_pixel":
        if mode == "nearest":
          exception.OP_UNSUPPORTED_EXCEPT(
              "Resize coordinate_transformation_mode=half_pixel " +
              "and mode=nearest", "Tensorflow")
        if mode == "cubic":
          if cubic_coeff_a == -0.75:
            exception.OP_UNSUPPORTED_EXCEPT(
                "Resize coordinate_transformation_mode=half_pixel, " +
                "mode=cubic and cubic_coeff_a=-0.75", "Tensorflow")
          if exclude_outside == 0:
            exception.OP_UNSUPPORTED_EXCEPT(
                "Resize coordinate_transformation_mode=half_pixel, " +
                "mode=cubic and exclude_outside=0", "Tensorflow")
        if len(node.inputs) == 3:  # sizes is not defined
          exception.OP_UNSUPPORTED_EXCEPT(
              "Resize coordinate_transformation_mode=half_pixel with scales",
              "Tensorflow")
      else:  # coordinate_transformation_mode == tf_crop_and_resize
        if mode == "nearest":
          if nearest_mode in ["round_prefer_floor", "ceil", "floor"]:
            exception.OP_UNSUPPORTED_EXCEPT(
                "Resize coordinate_transformation_mode=tf_crop_and_resize " +
                "and nearest_mode=" + nearest_mode, "Tensorflow")
        if mode == "cubic":
          exception.OP_UNSUPPORTED_EXCEPT(
              "Resize coordinate_transformation_mode=tf_crop_and_resize " +
              "and mode=cubic", "Tensorflow")

  @classmethod
  def version_10(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = tf_shape(x)
    scales = kwargs["tensor_dict"][node.inputs[1]]

    n_in_scales_is_one = tf.equal(scales[0], 1)
    c_in_scales_is_one = tf.logical_or(tf.equal(scales[1], 1),
                                       tf.equal(scales[3], 1))
    assert_n_c_in_scales_are_ones = tf.Assert(
        tf.logical_and(n_in_scales_is_one, c_in_scales_is_one), [scales])

    with tf.control_dependencies([assert_n_c_in_scales_are_ones]):
      x_in_NCHW_format = tf.equal(scales[1], 1)
      h_w_scale = tf.where(x_in_NCHW_format, scales[2:], scales[1:3])
      h_w_shape = tf.where(x_in_NCHW_format, x_shape[2:], x_shape[1:3])
      new_h_w_shape = tf.cast(h_w_scale * tf.cast(h_w_shape, scales.dtype),
                              tf.int32)

      mode = node.attrs.get("mode", "nearest")
      if mode.lower() == "linear":
        mode = tf.image.ResizeMethod.BILINEAR
      else:
        mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR

      def process_NCHW_format(x):
        x_t = tf.transpose(x, perm=[0, 2, 3, 1])
        y = tf.image.resize(x_t, size=new_h_w_shape, method=mode)
        y_t = tf.transpose(y, perm=[0, 3, 1, 2])
        return y_t

      def process_NHWC_format(x):
        y = tf.image.resize(x, size=new_h_w_shape, method=mode)
        return y

      output = tf.cond(x_in_NCHW_format, lambda: process_NCHW_format(x),
                       lambda: process_NHWC_format(x))

      return [output]

  @classmethod
  def version_11(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    x_shape = tf_shape(x)
    roi = tensor_dict[node.inputs[1]]
    scales = tensor_dict[node.inputs[2]]
    sizes = tensor_dict[node.inputs[3]] if len(
        node.inputs) == 4 else tf.constant([], tf.int64)
    coordinate_transformation_mode = node.attrs.get(
        "coordinate_transformation_mode", "half_pixel")
    cubic_coeff_a = node.attrs.get("cubic_coeff_a", -0.75)
    exclude_outside = node.attrs.get("exclude_outside", 0)
    extrapolation_value = node.attrs.get("extrapolation_value", 0.0)
    mode = node.attrs.get("mode", "nearest")
    nearest_mode = node.attrs.get("nearest_mode", "round_prefer_floor")

    param = tf.cond(tf.greater(tf.size(scales), 0), lambda: scales,
                    lambda: tf.cast(sizes, tf.float32))
    n_in_param_is_one = tf.equal(param[0], 1)
    c_in_param_is_one = tf.logical_or(tf.equal(param[1], 1),
                                      tf.equal(param[3], 1))
    assert_n_c_in_param_are_ones = tf.Assert(
        tf.logical_and(n_in_param_is_one, c_in_param_is_one), [param])

    with tf.control_dependencies([assert_n_c_in_param_are_ones]):
      if mode.lower() == "linear":
        mode = tf.image.ResizeMethod.BILINEAR
      elif mode.lower() == "cubic":
        mode = tf.image.ResizeMethod.BICUBIC
      else:
        mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR
      x_in_NCHW_format = tf.equal(param[1], 1)
      crop_and_resize = True if coordinate_transformation_mode == "tf_crop_and_resize" else False

      if len(node.inputs) == 3: # only scales is defined
        h_w_scale = tf.where(x_in_NCHW_format, scales[2:], scales[1:3])
        h_w_shape = tf.where(x_in_NCHW_format, x_shape[2:], x_shape[1:3])
        new_size = tf.cast(h_w_scale * tf.cast(h_w_shape, scales.dtype), tf.int32)
      else: # sizes is defined
        new_size = tf.cast(tf.where(x_in_NCHW_format, sizes[2:], sizes[1:3]), tf.int32)
      # Tensorflow require the shape of "size" in the "tf.image.resize" must be known at
      # graph creation time. However in the dynamic shape situation, the shape of "new_size"
      # will be "None", the actual shape can only be determine at runtime. But we know 
      # "new_size" should always contain [h, w], therefore the shape must be 2.
      new_size.set_shape([2])

      def get_NCHW_boxes():
        boxes = []
        if crop_and_resize:
          indices = []
          x_rank = len(x.get_shape())
          for i in range(2, x_rank):
            indices.insert(i - 2, i)
            indices.insert(i, i + x_rank)
          boxes = tf.expand_dims(tf.gather(roi, indices, axis=0), 0)
        return boxes

      def get_NHWC_boxes():
        boxes = []
        if crop_and_resize:
          indices = []
          x_rank = len(x.get_shape())
          for i in range(1, x_rank - 1):
            indices.insert(i - 1, i)
            indices.insert(i + 1, i + x_rank)
          boxes = tf.expand_dims(tf.gather(roi, indices, axis=0), 0)
        return boxes

      boxes = tf.cond(x_in_NCHW_format, get_NCHW_boxes, get_NHWC_boxes)
      box_indices = tf.cast(tf.range(0, x_shape[0]), dtype=tf.int32)

      def process_NCHW_format():
        x_t = tf.transpose(x, perm=[0, 2, 3, 1])
        if crop_and_resize:
          y = tf.image.crop_and_resize(x_t, boxes, box_indices, new_size, mode,
                                       extrapolation_value)
        else:
          y = tf.image.resize(x_t, size=new_size, method=mode)
        return tf.transpose(y, perm=[0, 3, 1, 2])

      def process_NHWC_format():
        if crop_and_resize:
          return tf.image.crop_and_resize(x, boxes, box_indices, new_size, mode,
                                          extrapolation_value)
        else:
          return tf.image.resize(x, size=new_size, method=mode)

      output = tf.cond(x_in_NCHW_format, process_NCHW_format,
                       process_NHWC_format)

      return [output]
