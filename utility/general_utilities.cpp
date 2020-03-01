/**
			  * @file				general_utilities.c
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Very general utilities needed: time couting, sorting, etc.
			  */

#include "general_utilities.h"


clock_t start;
char current_function[50];


int index_element(unsigned int* array, int length, unsigned int element)
{
  for(int i=0;i<length;i++){
    if(array[i] == element){
      return(i);
    }
  }
  return -1;
}


void start_crono(const char* function_name)
{

  clock_t start = clock();

  strcpy((char*)current_function, (char*)function_name);

  float seconds = (float)start / CLOCKS_PER_SEC;

  printf( HIGHLIGHTED"\n\n[STARTED");
  printf(BOLD " %s " RESET, current_function);
  printf(HIGHLIGHTED"AT %d SECS]\n" RESET, (int)seconds);
}


void stop_crono()
{

  clock_t end = clock();
  float seconds = (float)(end - start) / CLOCKS_PER_SEC;

  printf( HIGHLIGHTED"\n[ENDED");
  printf(BOLD " %s " RESET, current_function);
  printf(HIGHLIGHTED"AT %d SECS]\n\n" RESET, (int)seconds);
}


void print_info(char* message)
{
  fprintf(stdout, GREEN "\n\n\t[ %s ]\n\n" RESET, message);
}


void print_error(char* error)
{
  fprintf(stderr, RED "\n\n\t** %s **\n\n" RESET, error);
}


bool not_in(int element, unsigned int* array, int length_array)
{

  for(int i=0; i<length_array; i++){
    if(element == (int)array[i]){
      return false;
    }else if((int)array[i] == 0){
      break;
    }
  }

  return true;
}


void sort_array(unsigned int *A, int size)
{
	for(int i=0; i<size-1; i++)
	{
		int Imin = i;
		for(int j=i+1; j<size; j++)
		{
			if( A[j] < A[Imin] )
			{
				Imin = j;
			}
		}
		int temp = A[Imin];
		A[Imin] = A[i];
		A[i] = temp;
	}
}


int most_frequent_element(unsigned int *arr, int n)
{
    // Sort the array
    sort_array(arr, n);

    // find the max frequency using linear traversal
    int max_count = 1, res = arr[0], curr_count = 1;
    for (int i = 1; i < n; i++) {
        if (arr[i] == arr[i - 1])
            curr_count++;
        else {
            if (curr_count > max_count) {
                max_count = curr_count;
                res = arr[i - 1];
            }
            curr_count = 1;
        }
    }

    // If last element is most frequent
    if (curr_count > max_count)
    {
        max_count = curr_count;
        res = arr[n - 1];
    }

    return res;
}


int* force_integer_splits(int n, int x)
{
  int* parts = (int*)calloc(n, sizeof(int));
  if(x < n)
    parts[0] = -1;

  // If x % n == 0 then the minimum difference is 0 and all numbers are x / n
  else if (x % n == 0)
  {
      for(int i=0;i<n;i++){
        parts[i] = x/n;
      }
  }
  else{

      // upto n-(x % n) the values will be x / n after that the values will be x / n + 1
      int zp = n - (x % n);
      int pp = x/n;
      for(int i=0;i<n;i++)
      {
          if(i>= zp)
            parts[i] = pp + 1;
          else
            parts[i] = pp;
      }
  }
  return parts;
}


void exit_with_help()
{
	char message[1500] =
	"Usage: ./classification_scheme [hyperspectral image] [train reference data] [test reference data] [options]\n"
	"options:\n"
	"\t-s  -->  input_seg : input segmented image in RAW format | DEFAULT = segmentation algorithm applied to hyperspectral image\n"
	"\t-m  -->  output_clasfmap : output classification map | DEFAULT = ouput/map.ppm\n"
	"\t-f  -->  output_clasftxt : output classification textfile | DEFAULT = ouput/prediction.txt\n"
	"\t-p  -->  trainpredict_type : type of train and prediction procedure | DEFAULT = 3\n"
  "\t\t1 -- by pixel\n"
  "\t\t2 -- by blocks\n"
  "\t\t3 -- by segments\n"
	"\t-k  -->  kernel_type : SVM kernel type | DEFAULT = 0\n"
  "\t\t0 -- LINEAR kernel\n"
  "\t\t1 -- POLYNOMIAL kernel\n"
  "\t\t2 -- RBF kernel\n"
  "\t\t3 -- SIGMOID kernel\n"
	"\t-c  -->  C : set the parameter C of C-SVC | DEFAULT = 0.02\n"
	"\t-o  -->  output_model : output SVM model generated in train phase | DEFAULT = output/output.model\n"
	"\t-v  -->  verbose : set the quiet or verbose mode | DEFAULT = true\n"
	"\t-t  -->  texture_pipeline : texture algorithms (pipeline to use) | DEFAULT = 0\n"
  "\t\t0 -- no textures\n"
  "\t\t1 -- kmeans + vlad\n"
  "\t\t2 -- kmeans + bow\n"
  "\t\t3 -- gmm + fishervectors\n"
  "\t\t4 -- sift + km + vlad\n"
  "\t\t5 -- sift + gmm + fishervectors\n"
  "\t\t6 -- sift\n"
  "\t\t7 -- dsift + km + vlad\n"
  "\t\t8 -- dsift + gmm + fishervectors\n"
  "\t\t9 -- dsift\n"
  "\n\t * Parameters -t (1 or 2) and -p (any) are mutually exclusive\n";

  print_info((char*)message);
	exit(1);
}


void exit_input_error(int line_num)
{
  char error[100];
  sprintf(error, "Wrong input format at line %d", line_num);
  print_error((char*)error);
	exit(1);
}
