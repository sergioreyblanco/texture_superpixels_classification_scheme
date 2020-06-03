#!/bin/bash


> ../pruebas_china_class5.txt



images=()
while IFS= read -r line; do

        a=${line%".raw"}
        images+=( $a )

        echo "$a"

done < ../../../archivos_pesados/data/china/5class/Multis/names.txt






for im in "${images[@]}"
do

	echo -e "\n\n\n\t\t [${im}]\n" >> ../pruebas_china_class5.txt


	for i in {0..15}
	do

		echo -e "Metodo: ${i} " >> ../pruebas_china_class5.txt

		oa=`../texture_classification_scheme ../../../archivos_pesados/data/china/5class/Multis/${im}.raw ../../../archivos_pesados/data/china/5class/GTs/${im}_label_train.raw ../../../archivos_pesados/data/china/5class/GTs/${im}_label_test.raw -s ../../../archivos_pesados/data/china/5class/Segs/${im}_s1100.raw -t ${i} -4 0.1 17.5 -12 11 3 4 1 0.5 -5 20 8 0 -7 6 4 4 8 | grep ' OA:' | awk -F' ' '{print $3}'`
		
		echo -e "\tOA: ${oa}\n" >> ../pruebas_china_class5.txt
	done
done

