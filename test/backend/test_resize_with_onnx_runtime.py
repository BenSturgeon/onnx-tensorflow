import onnxruntime
import numpy as np
import onnx_tf

import onnxruntime.backend
from onnx import helper
from onnx import TensorProto

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]], dtype=np.float32)

def nearest_half_pixel_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      coordinate_transformation_mode='tf_half_pixel_for_nn',
      mode='nearest',
      nearest_mode='floor'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('coordinate_transformation_mode = tf_half_pixel_for_nn')
  print('nearest_mode = floor')
  print('data =')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output=')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print(format(err)) 

def nearest_half_pixel_sizes():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      coordinate_transformation_mode='tf_half_pixel_for_nn',
      mode='nearest',
      nearest_mode='floor'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 3, 3], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('coordinate_transformation_mode = tf_half_pixel_for_nn')
  print('nearest_mode = floor')
  print('data = ')
  print(data)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])

  np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])

def nearest_crop_and_resize_half_pixel_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      coordinate_transformation_mode='tf_crop_and_resize',
      mode='nearest',
      nearest_mode='round_prefer_ceil',
      extrapolation_value=-20.0
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

#  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('nearest_mode = round_prefer_ceil')
  print('coordinate_transformation_mode = tf_crop_and_resize')
  print('exclude_outside = 0')
  print('extrapolation_value = -20.0') 
  print('data =')
  print(data)
  print('roi = ', roi)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])
  print('shape = ', tf_output['Y'].shape)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])
  print('shape = ', rt_output[0].shape)

  np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])

def nearest_crop_and_resize_half_pixel_sizes():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      coordinate_transformation_mode='tf_crop_and_resize',
      mode='nearest',
      nearest_mode='round_prefer_ceil',
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
#  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 3, 3], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('nearest_mode = round_prefer_ceil')
  print('coordinate_transformation_mode = tf_crop_and_resize')
  print('exclude_outside = 0')
  print('extrapolation_value =00.0')
  print('data =')
  print(data)
  print('roi = ', roi)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])
  print('shape = ', tf_output['Y'].shape)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])
  print('shape = ', rt_output[0].shape)

  np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])

def linear_half_pixel_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='linear'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = linear')
  print('coordinate_transformation_mode = half_pixel')
  print('exclude_outside = 0')
  print('data =')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print(format(err))

def linear_half_pixel_sizes():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='linear'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 3, 3], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = linear')
  print('coordinate_transformation_mode = half_pixel')
  print('exclude_outside = 0')
  print('data =')
  print(data)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in linear_half_pixel_sizes')
    print(format(err))

def linear_crop_and_resize_half_pixel_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='linear',
      coordinate_transformation_mode='tf_crop_and_resize',
      extrapolation_value=20.0
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

#  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = linear')
  print('coordinate_transformation_mode = tf_crop_and_resize')
  print('exclude_outside = 0')
  print('data =')
  print(data)
  print('roi = ', roi)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])
  print('shape = ', tf_output['Y'].shape)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])
  print('shape = ', rt_output[0].shape)

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in linear_crop_and_resize_half_pixel_scales')
    print(format(err))

def linear_crop_and_resize_half_pixel_sizes():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='linear',
      coordinate_transformation_mode='tf_crop_and_resize',
      extrapolation_value=50.0
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
#  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 3, 3], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = linear')
  print('coordinate_transformation_mode = tf_crop_and_resize')
  print('exclude_outside = 0')
  print('data =')
  print(data)
  print('roi = ', roi)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])
  print('shape = ', tf_output['Y'].shape)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])
  print('shape = ', rt_output[0].shape)

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in linear_crop_and_resize_half_pixel_sizes')
    print(format(err))

  np.testing.assert_allclose(tf_output['Y'], rt_output[0], rtol=1e-6, atol=1e-6)  

def cubic_half_pixel_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='cubic',
      cubic_coeff_a=-0.5,
      exclude_outside=True
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = cubic')
  print('coordinate_transformation_mode = half_pixel')
  print('cubic_coeff_a = -0.5')
  print('exclude_outside = 1')
  print('data =')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print(format(err))

def cubic_half_pixel_size():
  node_def = helper.make_node(
      "Resize", 
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'], 
      mode='cubic',
      cubic_coeff_a=-0.5,
      exclude_outside=True
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 11)])

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 3, 3], dtype=np.int64)
  
  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = cubic')
  print('coordinate_transformation_mode = half_pixel')
  print('cubic_coeff_a = -0.5')
  print('exclude_outside = 1')
  print('data =')
  print(data)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in cubic_half_pixel_size')
    print(format(err))

  np.testing.assert_allclose(tf_output['Y'], rt_output[0], rtol=1e-3, atol=1e-3)
  
  

def main():
#  nearest_half_pixel_scales()  
  nearest_half_pixel_sizes()
#  linear_half_pixel_scales()
  linear_half_pixel_sizes()
#  cubic_half_pixel_scales()
  cubic_half_pixel_size()
  nearest_crop_and_resize_half_pixel_scales()
  nearest_crop_and_resize_half_pixel_sizes()
  linear_crop_and_resize_half_pixel_scales()
  linear_crop_and_resize_half_pixel_sizes()

if __name__ == '__main__':
  main()
