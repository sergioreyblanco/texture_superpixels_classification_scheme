#!/bin/bash


> ../pruebas_optm_liop.txt
list=()
images=(Salinas Indian Pavia z1 Mestas Emb Fer) #Salinas Indian Pavia z1 Mestas Emb Fer
seg_sizes=(10 10 10 1100 1100 1100 1100) #10 10 10 1100 1100 1100 1100
seg_sizes_index=0


function max_float {
	max=0.0
	idmax=0
	index=0
	for i in "${list[@]}"
	do
		if (( $(echo "$i > $max" |bc -l) )); then
			max=$i
			idmax=$index
		fi
		((index++))
	done
}







for im in "${images[@]}"
do
        echo -e "$im"
        echo -e "\n\n\n\t ${im}\n" >> ../pruebas_optm_liop.txt

	#inicial --> 20 4 4 2 0.5
	oa=`../texture_classification_scheme ../../../archivos_pesados/data/${im}_multi.raw ../../../archivos_pesados/data/${im}_train.raw ../../../archivos_pesados/data/${im}_test.raw -s ../../../archivos_pesados/data/${im}_s${seg_sizes[$seg_sizes_index]}.raw -t 14 -12 20 4 4 2 0.5 | grep ' OA:' | awk -F' ' '{print $3}'`
	echo -e "\t\t inicial: $oa" >> ../pruebas_optm_liop.txt

	list=()
	patch_size=0.0
	neighbours=0.0
	bins=0.0
	size_of_radius=0.0
	intensity_threshold=0.0

	#30 30 25 22 21 20 19 15 10 9 8 7
	patch_size_list=()
	patch_size_list=(30 30 25 22 21 20 19 15 10 9 8 7)
	for i in "${patch_size_list[@]}"
	do
	  echo -e "$i"
	  oa=`../texture_classification_scheme ../../../archivos_pesados/data/${im}_multi.raw ../../../archivos_pesados/data/${im}_train.raw ../../../archivos_pesados/data/${im}_test.raw -s ../../../archivos_pesados/data/${im}_s${seg_sizes[$seg_sizes_index]}.raw -t 14 -12 $i 4 4 2 0.5 | grep ' OA:' | awk -F' ' '{print $3}'`
	  list+=( $oa )
	done

	max_float
	patch_size=${patch_size_list[$idmax]}
	echo -e "\t\t patch_size: $patch_size -> $max" >> ../pruebas_optm_liop.txt

	################################################

	list=()
	#3 3 4 5
	neighbours_list=()
	neighbours_list=(3 3 4 5)
	for i in "${neighbours_list[@]}"
	do
	  echo -e "$i"
	  oa=`../texture_classification_scheme ../../../archivos_pesados/data/${im}_multi.raw ../../../archivos_pesados/data/${im}_train.raw ../../../archivos_pesados/data/${im}_test.raw -s ../../../archivos_pesados/data/${im}_s${seg_sizes[$seg_sizes_index]}.raw -t 14 -12 $patch_size $i 4 2 0.5 | grep ' OA:' | awk -F' ' '{print $3}'`
	  list+=( $oa )
	done

	max_float
	neighbours=${neighbours_list[$idmax]}
	echo -e "\t\t neighbours: $neighbours -> $max" >> ../pruebas_optm_liop.txt

	################################################

	list=()
	#3 3 4 5
	bins_list=()
	bins_list=(3 3 4 5)
	for i in "${bins_list[@]}"
	do
	  echo -e "$i"
	  oa=`../texture_classification_scheme ../../../archivos_pesados/data/${im}_multi.raw ../../../archivos_pesados/data/${im}_train.raw ../../../archivos_pesados/data/${im}_test.raw -s ../../../archivos_pesados/data/${im}_s${seg_sizes[$seg_sizes_index]}.raw -t 14 -12 $patch_size $neighbours $i 2 0.5 | grep ' OA:' | awk -F' ' '{print $3}'`
	  list+=( $oa )
	done

	max_float
	bins=${bins_list[$idmax]}
	echo -e "\t\t bins: $bins -> $max" >> ../pruebas_optm_liop.txt

	################################################

	list=()
	#1 1 2 6 9 10 11
	# 1 1 2 3 4 5 6
	size_of_radius_list=()
	size_of_radius_list=(1 1 2 6 11 12 13)
	for i in "${size_of_radius_list[@]}"
	do
	  echo -e "$i"
	  oa=`../texture_classification_scheme ../../../archivos_pesados/data/${im}_multi.raw ../../../archivos_pesados/data/${im}_train.raw ../../../archivos_pesados/data/${im}_test.raw -s ../../../archivos_pesados/data/${im}_s${seg_sizes[$seg_sizes_index]}.raw -t 14 -12 $patch_size $neighbours $bins $i 0.5 | grep ' OA:' | awk -F' ' '{print $3}'`
	  list+=( $oa )
	done

	max_float
	size_of_radius=${size_of_radius_list[$idmax]}
	echo -e "\t\t size_of_radius: $size_of_radius -> $max" >> ../pruebas_optm_liop.txt

	################################################

	list=()
	intensity_threshold_list=()
	for i in $(seq 0 0.1 1); do intensity_threshold_list+=( $i ) ; done
	for i in "${intensity_threshold_list[@]}"
	do
	  j=`sed 's/\,/\./' <<< $i`
	  echo -e "$j"
	  oa=`../texture_classification_scheme ../../../archivos_pesados/data/${im}_multi.raw ../../../archivos_pesados/data/${im}_train.raw ../../../archivos_pesados/data/${im}_test.raw -s ../../../archivos_pesados/data/${im}_s${seg_sizes[$seg_sizes_index]}.raw -t 14 -12 $patch_size $neighbours $bins $size_of_radius $j | grep ' OA:' | awk -F' ' '{print $3}'`
	  list+=( $oa )
	done

	max_float
	intensity_threshold=${intensity_threshold_list[$idmax]}
	echo -e "\t\t intensity_threshold: $intensity_threshold -> $max" >> ../pruebas_optm_liop.txt

	echo -e "\t final: [ $patch_size $neighbours $bins $size_of_radius $intensity_threshold ]" >> ../pruebas_optm_liop.txt

	((seg_sizes_index++))
done

