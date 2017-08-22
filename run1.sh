for i in 'stcu' 'stla' 'stre' 'rtcu' 'rtla' 'rtre' 'last' 'lart' 'lare' 'lacu' 'curt' 'cula' 'cure' 'cust' 'rert' 'rela' 'recu' 'rest'
do
  echo RDASA_MT_4_${i}.hdf5
  th main_dasa.lua -data preprocess\ codes/hdf5file/RDASA_MT_4_${i}.hdf5 -cudnn 0
done