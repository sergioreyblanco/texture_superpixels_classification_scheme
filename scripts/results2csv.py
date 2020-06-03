#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Converts the results of OA got from the scripts to csv format

Examples:
	pruebas clase 15: CITIUS_privado/texturas/algoritmos_adaptados/texture_classification_scheme/scripts/results2csv.py -t china_general -i CITIUS_privado/texturas/pruebas/pruebas_china_class15.txt -o CITIUS_privado/texturas/pruebas/pruebaChina_class15.xlsx -w excel

	pruebas clase 15 BOW (poner ["BOW"] en .index): CITIUS_privado/texturas/algoritmos_adaptados/texture_classification_scheme/scripts/results2csv.py -t china_general -i CITIUS_privado/texturas/pruebas/pruebas_china_class15_bow.txt -o CITIUS_privado/texturas/pruebas/pruebaChina_class15_bow.xlsx -w excel

	pruebas clase 5: CITIUS_privado/texturas/algoritmos_adaptados/texture_classification_scheme/scripts/results2csv.py -t china_general -i CITIUS_privado/texturas/pruebas/pruebas_china_class5.txt -o CITIUS_privado/texturas/pruebas/pruebaChina_class5.xlsx -w excel

	pruebas clase 5 bow (poner ["BOW"] en .index): CITIUS_privado/texturas/algoritmos_adaptados/texture_classification_scheme/scripts/results2csv.py -t china_general -i CITIUS_privado/texturas/pruebas/pruebas_china_class5_bow.txt -o CITIUS_privado/texturas/pruebas/pruebaChina_class5_bow.xlsx -w excel

	pruebas rios e hiper: CITIUS_privado/texturas/algoritmos_adaptados/texture_classification_scheme/scripts/results2csv.py -t hiperYrios -i CITIUS_privado/texturas/pruebas/pruebas_hiperYrios.txt -o CITIUS_privado/texturas/pruebas/pruebas_hiperYrios.xlsx -w excel

	pruebas rios e hiper BOW (CAMBIAR): CITIUS_privado/texturas/algoritmos_adaptados/texture_classification_scheme/scripts/results2csv.py -t hiperYrios -i CITIUS_privado/texturas/pruebas/pruebas_hiperYrios_bow.txt -o CITIUS_privado/texturas/pruebas/pruebas_hiperYrios_bow.xlsx -w excel

	pruebas china clase 15 graph: CITIUS_privado/texturas/algoritmos_adaptados/texture_classification_scheme/scripts/results2csv.py -t china_graph -i CITIUS_privado/texturas/pruebas/pruebas_china_class15.txt -o CITIUS_privado/texturas/pruebas/pruebaChina_class15_graph.xlsx -w excel

	pruebas china clase 5 graph: CITIUS_privado/texturas/algoritmos_adaptados/texture_classification_scheme/scripts/results2csv.py -t china_graph -i CITIUS_privado/texturas/pruebas/pruebas_china_class5.txt -o CITIUS_privado/texturas/pruebas/pruebaChina_class5_graph.xlsx -w excel

"""

import sys, getopt
import pandas as pd
import re
from pprint import pprint
from openpyxl import workbook
import numpy as np

method_names_china=["WithoutText", "Kmeans+BOW", "Kmeans+VLAD", "GMM+FV",
		"SIFT+Kmeans+VLAD (descs)","SIFT+GMM+FV (descs)","DSIFT+Kmeans+VLAD (descs)",
		"DSIFT+GMM+FV (descs)","LIOP+Kmeans+VLAD","HOG+Kmeans+VLAD"]

method_names_general=["WithoutText", "Kmeans+VLAD", "Kmeans+BOW", "GMM+FV",
                "SIFT+Kmeans+VLAD (descs)","SIFT+GMM+FV (descs)",
                "DSIFT+Kmeans+VLAD (descs)", "DSIFT+GMM+FV (descs)",
                "LIOP+Kmeans+VLAD (no descs)","LIOP+GMM+FV (no descs)",
                "HOG+Kmeans+VLAD (no descs)","HOG+GMM+FV (no descs)"]

param_liop_names=["Initial", "patch_size", "neighbours", "bins", "size_of_radius", "intensity_threshold", "Final"]

param_hog_names=["Initial", "numOrientations", "cellSize", "bilinearOrientationAssingments", "Final"]

method_trains=["Kmeans+BoW", "GMM+FV", "DSIFT+GMM+FV (descs)", "LIOP+Kmeans+VLAD (no descs)"]


def parse_results(type, path):

   file = open(path, 'r')
   Lines = file.readlines() 



   if type == 'rios_completo':
      images=re.compile('\t{2} {1}')
      col_names=[]
      pipelines=re.compile('\t METODO: ((0)|(1)|(2)|(3)|(6)|(7)|(10)|(11)|(12)|(13)|(14)|(15))')
      selected_method = False
      oa=re.compile('\t  OA:')
      aa=re.compile('.*AA:')
      kappa=re.compile('\t  Kappa:')
      time=re.compile('.*ENDED.*TEST.*')
      data=[]
      c1=[]
      c2=[]
      c3=[]
      c4=[]

      for line in Lines:
         if images.match(line) != None and c1 == []:
             #print("*0*", line[3 : 3 + len(line)-4])
             col_names.append(line[3 : 3 + len(line)-4])
         elif images.match(line) != None and c1 != []:
             #data.append(row)
             #row=[]
             data.append(c1)
             c1=[]
             data.append(c2)
             c2=[]
             data.append(c3)
             c3=[]
             data.append(c4)
             c4=[]
             col_names.append(line[3 : 3 + len(line)-4])
         elif pipelines.match(line) != None:
             #print("*1*", line)
             selected_method = True
         elif oa.match(line) != None and selected_method == True:
             #print("*2*", line[13 : 13 + len(line)-16])
             c1.append(float(line[13 : 13 + len(line)-16]))
         elif aa.match(line) != None and selected_method == True:
             #print("*3*", line[18 : 18 + len(line)-21])
             c2.append(float(line[18 : 18 + len(line)-21]))
         elif kappa.match(line) != None and selected_method == True:
             #print("*4*", line[11 : 11 + len(line)-16])
             c3.append(float(line[11 : 11 + len(line)-16]))
         elif time.match(line) != None and selected_method == True:
             #print("*5*", line[27 : 27 + len(line)-16].split(" ")[0])
             c4.append(int(line[27 : 27 + len(line)-16].split(" ")[0]))
             selected_method = False

      #data.append(row)
      #row=[]
      data.append(c1)
      c1=[]
      data.append(c2)
      c2=[]
      data.append(c3)
      c3=[]
      data.append(c4)
      c4=[]

      df= pd.DataFrame(data)
      df = df.transpose()
      #df.columns = col_names
      df.index = method_names_general

      #df["Mean"] = df.mean(axis=1)

      pprint(df)


   if type == 'hiperYrios':
      images=re.compile('\t{2} {1}')
      col_names=[]
      pipelines=re.compile('\t METODO: ((0)|(2)|(1)|(3)|(6)|(7)|(10)|(11)|(12)|(14))')
      selected_method = False
      oa=re.compile('\t  OA:')
      data=[]
      row=[]
      for line in Lines:
         if pipelines.match(line) != None and selected_method == True:
            row.append(np.nan)

         elif images.match(line) != None and row == []:
            col_names.append(line[3 : 3 + len(line)-4])

         elif images.match(line) != None and row != []:
            data.append(row)
            row=[]
            col_names.append(line[3 : 3 + len(line)-4])

         elif pipelines.match(line) != None:
            selected_method = True

         elif oa.match(line) != None and selected_method == True:
            row.append(float(line[13 : 13 + len(line)-16]))
            selected_method = False

      data.append(row)
      row=[]

      df= pd.DataFrame(data)
      df = df.transpose()
      df.columns = col_names
      df.index = ["BOW"] #["BOW"] method_names_general

      df["Mean"] = df.mean(axis=1)
      df["SDev"] = df.std(axis=1)

      pprint(df)


   if type == 'trains':
      percentages=re.compile('\t{1} {1}')
      images=re.compile('\t{2} {1}')
      methods=re.compile('\t{3} {1}[^Seed].*')
      seeds=re.compile('\t{3} {1}Seed')
      col_names_images = []
      row_names_percentages = []
      cell_names_methods = []
      cell_names_seeds = []
      cell_oas = []

      for line in Lines:
         if percentages.match(line):
            #print("1* ", line.split(" ")[2][0 : 0 + len(line.split(" ")[2])-1])
            cur_percentage = line.split(" ")[2][0 : 0 + len(line.split(" ")[2])-1]
            if cur_percentage not in row_names_percentages:
                row_names_percentages.append(cur_percentage)

         elif images.match(line):
            #print("2* ", line.split(" ")[2][0 : 0 + len(line.split(" ")[2])-1])
            cur_image = line.split(" ")[2][0 : 0 + len(line.split(" ")[2])-1]
            if cur_image not in col_names_images:
                col_names_images.append(cur_image)

         elif methods.match(line):
            #print("3a* ", line.split(":")[0][4 : 4 + len(line.split(":")[0])-0])
            cur_method = line.split(":")[0][4 : 4 + len(line.split(":")[0])-0]
            if cur_method not in cell_names_methods:
                cell_names_methods.append(cur_method)

            cell_oas.append( line.split(":")[1][1 : 1 + len(line.split(":")[1])-2] )

         elif seeds.match(line):
            #print("4* ", line.split(" ")[2][0 : 0 + len(line.split(" ")[2])-1])
            cur_seed = line.split(" ")[2][0 : 0 + len(line.split(" ")[2])-1]
            if cur_seed not in cell_names_seeds:
                cell_names_seeds.append(cur_seed)

      #print(col_names_images)
      #print(row_names_percentages)
      #print(len(cell_oas))
      #print(cell_names_methods)
      #print(cell_names_seeds)
      #print(cell_oas)

      data=[]
      row=[]
      cell_index=[]
      cell_values=[]
      oa_index=0
      for i in range(len(row_names_percentages)):
          for j in range(len(col_names_images)):
              for a in range(len(cell_names_methods)):
                  for b in range(len(cell_names_seeds)):
                     cell_index.append((a+b*len(cell_names_methods))+oa_index)
              for k in range(len(cell_index)):
                  cell_values.append(cell_oas[cell_index[k]])
              #print(cell_values)
              v=""
              aux=[]
              for c in range(len(cell_names_methods)):
                 v += cell_names_methods[c] + ": \n"
                 for d in range(len(cell_names_seeds)):
                     #v += "\t\n seed " + cell_names_seeds[d] + " $\\xrightarrow{}$ " + str(round(float(cell_values[c*len(cell_names_seeds)+d]), 2))
                     aux.append(float(cell_values[c*len(cell_names_seeds)+d]))

                 v+= "mean -> " + str(round(np.mean(aux), 2)) + " \n"
                 v+= "sdev -> " + str(round(np.std(aux), 2)) + " \n"
                 v+="\n\n\n"
              row.append(v)
              cell_index=[]
              cell_values=[]
              oa_index = oa_index + (len(cell_names_methods)*len(cell_names_seeds))

          data.append(row)
          #print("a")
          #print(row)
          row=[]

      df= pd.DataFrame(data)
      df.columns = col_names_images
      df.index = row_names_percentages

      pprint(df)



   if type == 'optm_hog':
      images=re.compile('\t{1} {1}')
      final=re.compile('\t{1} {1}(final:)')
      col_names=[]
      params=re.compile('\t{2} {1}')
      data=[]
      row=[]
      flag=False
      for line in Lines:
         if images.match(line) != None and final.match(line) ==None and row == []:
            flag=True
            col_names.append(line[2 : 2 + len(line)-3])

         elif params.match(line) != None:
            if flag==True:
                flag=False
                oa = line.split(" ")[2]
                param = "";
                row.append(param+str(round(float(oa[0 : 0 + len(oa)-1]), 2)))
            else:
                oa = line.split(" ")[4]
                param = line.split(" ")[2]
                row.append(param[0 : 0 + len(param)] + " $\\xrightarrow{}$ " + str(round(float(oa[0 : 0 + len(oa)-1]),2)))

         elif final.match(line) != None:
            f = line.split("[")[1]
            row.append("[" + f[1 : 1 + len(f)-4] + "]")
            data.append(row)
            row=[]

      df= pd.DataFrame(data)
      df = df.transpose()
      df.columns = col_names
      df.index = param_hog_names

      pprint(df)


   if type == 'optm_liop':
      images=re.compile('\t{1} {1}')
      final=re.compile('\t{1} {1}(final:)')
      col_names=[]
      params=re.compile('\t{2} {1}')
      data=[]
      row=[]
      flag=False
      for line in Lines:
         if images.match(line) != None and final.match(line) ==None and row == []:
            flag=True
            col_names.append(line[2 : 2 + len(line)-3])

         elif params.match(line) != None:
            if flag==True:
                flag=False
                oa = line.split(" ")[2]
                param = ""
                row.append(param+str(round(float(oa[0 : 0 + len(oa)-1]),2)))
            else:
                oa = line.split(" ")[4]
                param = line.split(" ")[2]
                row.append(param[0 : 0 + len(param)] + " $\\xrightarrow{}$ " + str(round(float(oa[0 : 0 + len(oa)-1]),2)))

         elif final.match(line) != None:
            f = line.split("[")[1]
            row.append("[" + f[1 : 1 + len(f)-4] + "]")
            data.append(row)
            row=[]

      df= pd.DataFrame(data)
      df = df.transpose()
      df.columns = col_names
      df.index = param_liop_names

      pprint(df)


   if type == 'hiperYrios':
      images=re.compile('\t{2} {1}')
      col_names=[]
      pipelines=re.compile('\t METODO: ((0)|(2)|(1)|(3)|(6)|(7)|(10)|(11)|(12)|(14))')
      selected_method = False
      oa=re.compile('\t  OA:')
      data=[]
      row=[]
      for line in Lines:
         if pipelines.match(line) != None and selected_method == True:
            row.append(np.nan)

         elif images.match(line) != None and row == []:
            col_names.append(line[3 : 3 + len(line)-4])

         elif images.match(line) != None and row != []:
            data.append(row)
            row=[]
	    col_names.append(line[3 : 3 + len(line)-4])

         elif pipelines.match(line) != None:
            selected_method = True

         elif oa.match(line) != None and selected_method == True:
            row.append(float(line[13 : 13 + len(line)-16]))
            selected_method = False

      data.append(row)
      row=[]

      df= pd.DataFrame(data)
      df = df.transpose()
      df.columns = col_names
      df.index = ["BOW"] #["BOW"] method_names_general

      df["Mean"] = df.mean(axis=1)
      df["SDev"] = df.std(axis=1)

      pprint(df)

   if type == 'china_general':
      data=[]
      images=re.compile('[\t ]*\[.*\]')
      pipelines=re.compile('Metodo\: ((0 )|(1 )|(2 )|(3 )|(6 )|(7 )|(10 )|(11 )|(12 )|(13 )|(14 )|(15 ))')
      oa=re.compile('\tOA: ([0-9]|\.)+')
      oa_failed=re.compile('\tOA: \n')
      row=[]
      col_names=[]
      selected_method = False
      for line in Lines:

         if images.match(line) != None:
            col_names.append(line[4 : 4 + len(line)-6])

         if images.match(line) != None and row != []:
            data.append(row)
            row=[]

         elif pipelines.match(line) != None:
            selected_method = True

         elif oa_failed.match(line) != None and selected_method == True:
            row.append(np.nan)
            selected_method = False

         elif oa.match(line) != None and selected_method == True:
            row.append(float(line.split(" ")[1][0:0+7]))
            selected_method = False

      data.append(row)
      row=[]

      df= pd.DataFrame(data)
      df = df.transpose()
      df.columns = col_names
      df.index = method_names_general #["BOW"] method_names_general

      df["Mean"] = df.mean(axis=1)
      df["SDev"] = df.std(axis=1)

      pprint(df)



   if type == 'china_graph':
      data=[]
      images=re.compile('[\t ]*\[.*\]')
      pipelines=re.compile('Metodo\: ((0 )|(2 )|(1 )|(3 )|(6 )|(7 )|(10 )|(11 )|(12 )|(14 ))')
      oa=re.compile('\tOA: ([0-9]|\.)+')
      oa_failed=re.compile('\tOA: \n')
      row=[]
      col_names=[]
      selected_method = False
      for line in Lines:

         if images.match(line) != None:
            col_names.append(line[4 : 4 + len(line)-6])

         if images.match(line) != None and row != []:
            data.append(row)
            row=[]

         elif pipelines.match(line) != None:
            selected_method = True

         elif oa_failed.match(line) != None and selected_method == True:
            row.append(np.nan)
            selected_method = False

         elif oa.match(line) != None and selected_method == True:
            row.append(float(line.split(" ")[1][0:0+7]))
            selected_method = False

      data.append(row)
      row=[]

      df= pd.DataFrame(data)
      df = df.transpose()
      df.columns = col_names
      df.index = method_names_china

      df["Mean"] = df.mean(axis=1)

      pprint(df)

   return(df)



def write2csv(df, path):
   df.to_csv(path)



def write2excel(df, path):

   df.to_excel(path)






def main(argv):

   type = ''
   ifile = ''
   ofile = ''
   wformat = ''
   try:
      opts, args = getopt.getopt(argv,"ht:i:o:w:",["type=","ifile=","ofile=","wformat="])
   except getopt.GetoptError:
      print 'results2csv.py -t <china,rios> -i <input_path> -o <output_path> -w <write_format>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'results2csv.py -t <china,rios> -i <input_path> -o <output_path>'
         sys.exit()
      elif opt in ("-t", "--type"):
         type = arg
      elif opt in ("-i", "--ifile"):
         ifile = arg
      elif opt in ("-o", "--ofile"):
         ofile = arg
      elif opt in ("-w", "--wformat"):
         wformat = arg

   df = parse_results(type, ifile)
   print('Parsing finished')

   if wformat == 'csv':
       write2csv(df, ofile)
   elif wformat == 'excel':
       write2excel(df, ofile)
   print("Writing finished")



if __name__ == "__main__":
   main(sys.argv[1:])
