/**
			  * @file				general_utilities.c
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Very general utilities needed: time couting, sorting, etc.
			  */

#include "general_utilities.h"

struct timeval  tv1;
char current_function[50];


void find_maxmin(unsigned int *data, int numData, long long unsigned int* min_value, long long unsigned int* max_value)
{
  long ma = 0, mi=10000000;

  for(int i=0; i<numData;i++){
    if(data[i] > ma){
      ma = data[i];
    } else if(data[i] < mi){
      mi = data[i];
    }
  }

  (*max_value) = ma;
  (*min_value) = mi;
}


int factorial(int n)
{
    return (n==1 || n==0) ? 1: n * factorial(n - 1);
}


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


  strcpy((char*)current_function, (char*)function_name);

  struct timeval  tv2;
  gettimeofday(&tv2, NULL);

  printf( HIGHLIGHTED"\n\n[STARTED");
  printf(BOLD " %s " RESET, current_function);
  long elapsed = (tv2.tv_sec-tv1.tv_sec);
  printf(HIGHLIGHTED"AT %d SECS]\n" RESET, (int) elapsed);
}


void stop_crono()
{


  struct timeval  tv2;
  gettimeofday(&tv2, NULL);


  printf( HIGHLIGHTED"\n[ENDED");
  printf(BOLD " %s " RESET, current_function);
  long elapsed = (tv2.tv_sec-tv1.tv_sec);
  printf(HIGHLIGHTED"AT %d SECS]\n\n" RESET, (int) (elapsed));
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

  print_info((char*)help_message);
	exit(1);
}


void exit_input_error(int line_num)
{
  char error[100];
  sprintf(error, "Wrong input format at line %d", line_num);
  print_error((char*)error);
	exit(1);
}
