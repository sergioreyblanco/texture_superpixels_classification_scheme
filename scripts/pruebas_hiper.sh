#!/bin/bash

> ../pruebas_hiper.txt

echo -e "\n\n\n\t\t IndianPines\n\n" >> ../pruebas_hiper.txt
for i in {12..15}
do
	echo -e "\n\n\n\n\t METODO: $i" >> ../pruebas_hiper.txt
	../texture_classification_scheme ../../../archivos_pesados/data/Indian_multi.raw ../../../archivos_pesados/data/Indian_train.raw ../../../archivos_pesados/data/Indian_test.raw -s ../../../archivos_pesados/data/Indian_s10.raw -t $i -4 0.1 2.5 -7 2 4 4 8 -5 20 8 0 -12 7 4 4 2 0.5 >> ../pruebas_hiper.txt
done

echo -e "\n\n\n\t\t Salinas\n\n" >> ../pruebas_hiper.txt
for i in {12..15}
do
        echo -e "\n\n\n\n\t METODO: $i" >> ../pruebas_hiper.txt
        ../texture_classification_scheme ../../../archivos_pesados/data/Salinas_multi.raw ../../../archivos_pesados/data/Salinas_train.raw ../../../archivos_pesados/data/Salinas_test.raw -s ../../../archivos_pesados/data/Salinas_s10.raw -t $i -4 0.1 2.5 -7 2 4 4 8 -5 28 2 0 -12 10 4 4 10 0.6 >> ../pruebas_hiper.txt
done

echo -e "\n\n\n\t\t Pavia\n\n" >> ../pruebas_hiper.txt
for i in {12..15}
do
	echo -e "\n\n\n\n\t METODO: $i" >> ../pruebas_hiper.txt
	../texture_classification_scheme ../../../archivos_pesados/data/Pavia_multi.raw ../../../archivos_pesados/data/Pavia_train.raw ../../../archivos_pesados/data/Pavia_test.raw -s ../../../archivos_pesados/data/Pavia_s10.raw -t $i -4 0.1 2.5 -7 2 4 4 8 -5 32 12 1 -12 15 4 4 2 0.5 >> ../pruebas_hiper.txt
done
