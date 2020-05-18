#!/bin/bash
dir_name="iPhone_7p"
echo $dir_name
if [ ! -d "./output_noise/$dir_name" ]; then
	mkdir ./output_noise/$dir_name
fi

for ((i=1; i<=2; i ++))
do
  save_path=$dir_name"/"$i
  cd $save_path
  echo $save_path
  if [ ! -d "../../output_noise/$save_path" ]; then
		mkdir ../../output_noise/$save_path
 	fi
  
  count=1
  for fullname in $(ls *.jpg)
  do
    filename=$(echo $fullname | sed 's/\.[^.]*$//')
    if [ $count -eq 1 ]; then
      cd ../../
    fi
    python test_ffdnet_ipol.py --save_path $save_path --input $save_path"/"$filename.jpg
    count=0
  done
done

