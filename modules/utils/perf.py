if __name__ == '__main__':
  import sys
  import sophon.sail as sail
  bmodel_path = sys.argv[1]
  try:
    tpu_id_list = list(map(int, sys.argv[2].split(',')))
  except:
    print('please check input tpu id format, e.g. 1,2,3')
    sys.exit(-1)
  tpu_count = sail.get_available_tpu_num()
  tpu_id_list = list(set(tpu_id_list))
  tpu_id_list = list(filter(lambda x: 0 <= x < tpu_count, tpu_id_list))
  print('---------- {} -----------'.format(tpu_id_list))
  sail._perf(bmodel_path, tpu_id_list)
