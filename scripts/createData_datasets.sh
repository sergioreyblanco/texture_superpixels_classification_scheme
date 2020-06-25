#!/bin/bash

squote="'"
list_trains=(0.20)
list_names=(z1 Mestas Fer Emb)
list_dsift=("-7 4 4 4 8" "-7 4 4 4 8" "-7 6 4 4 8" "-7 6 4 4 8")
list_liop=("-12 7 3 4 2 0.5" "-12 19 3 4 1 0.2" "-12 7 3 4 2 0.5" "-12 25 4 4 3 0.7")
list_sizes=(1100 1100 1100 1100)
list_seeds=(0 347 589)
seg_sizes_index=0



> ../pruebas_trains.txt



for i in "${list_trains[@]}"
do

	echo -e "\n\n\n\t Percentage ${i}\n" >> ../pruebas_trains.txt

	for j in "${list_names[@]}"
	do

		echo -e "\n\t\t Image ${j}\n" >> ../pruebas_trains.txt

		for k in "${list_seeds[@]}"
		do
		
			echo -e "\t\t\t Seed ${k}: " >> ../pruebas_trains.txt	

			./dataset_train.out ./${j}_gt.raw ./${j}_s${list_sizes[$seg_sizes_index]}.raw ${k} ${i}
			rm ./samples.csv


			command3="create_test_image("$squote"./"${j}"_gt.raw"$squote", "$squote"./"${j}"_train_${k}_${i}.raw"$squote");quit;"
			matlab -nodisplay -nodesktop -r "`echo "$command3"`";


			oa=`../../algoritmos_adaptados/texture_classification_scheme/texture_classification_scheme ${j}_multi.raw ${j}_train_${k}_${i}.raw ${j}_test_${k}_${i}.raw -s ${j}_s${list_sizes[$seg_sizes_index]}.raw -t 2 | grep ' OA:' | awk -F' ' '{print $3}'`
			echo -e "\t\t\t Kmeans+BOW: $oa\n" >> ../pruebas_trains.txt


			oa=`../../algoritmos_adaptados/texture_classification_scheme/texture_classification_scheme ${j}_multi.raw ${j}_train_${k}_${i}.raw ${j}_test_${k}_${i}.raw -s ${j}_s${list_sizes[$seg_sizes_index]}.raw -t 3 | grep ' OA:' | awk -F' ' '{print $3}'`
			echo -e "\t\t\t GMM+FisherVectors: $oa\n" >> ../pruebas_trains.txt


			oa=`../../algoritmos_adaptados/texture_classification_scheme/texture_classification_scheme ${j}_multi.raw ${j}_train_${k}_${i}.raw ${j}_test_${k}_${i}.raw -s ${j}_s${list_sizes[$seg_sizes_index]}.raw -t 11 ${list_dsift[$seg_sizes_index]} | grep ' OA:' | awk -F' ' '{print $3}'`
			echo -e "\t\t\t DSIFT+GMM+FisherVectors (descs): $oa\n" >> ../pruebas_trains.txt


			oa=`../../algoritmos_adaptados/texture_classification_scheme/texture_classification_scheme ${j}_multi.raw ${j}_train_${k}_${i}.raw ${j}_test_${k}_${i}.raw -s ${j}_s${list_sizes[$seg_sizes_index]}.raw -t 12 ${list_liop[$seg_sizes_index]} | grep ' OA:' | awk -F' ' '{print $3}'`
			echo -e "\t\t\t LIOP+Kmeans+VLAD (no descs): $oa\n" >> ../pruebas_trains.txt

		done

		((seg_sizes_index++))

	done
	seg_sizes_index=0
done















