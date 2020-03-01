
/**
			  * @file				bow.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Bag of Words algorithm for texture descriptors computation.
			  */

#include "bow.h"

//----------------------------------------------------------------------------
// UTIL, SELECCION MUESTRAS, CALCULO DE PRECISIONES
//----------------------------------------------------------------------------

// fisher-yates shuffle
void random_shuffle_bow(int *a, int N)
{  int i, j, t;
   for(i=N-1;i>0;i--)
   {  j=(int)drand48()*(i+1);
      t=a[j]; a[j]=a[i]; a[i]=t; }
}


// Selecciona un conjunto ejemplo de muestras de entrenamiento de acuerdo con los requerimientos
// Entrada: ngtc: numero de muestras por clase en el gt
// Entrada: ntrc: numero de muestras por clase que se usaran en el entrenamiento
// Entrada: listc[clases][muestras]: enumera las muestras del gt por clase y en formato y*H+x
// Salida: index: enumera las muestras que se utilizaran en el entrenamiento en el formato y*H+x
// salida: maptr: mapa que para cada pixel contiene 1=training, 2=testing, 0=ninguno de los dos
//         para la clasificacion de la imagen completa solo se usan los de training (bits 1)
void select_training_samples_bow(unsigned char *truth, int *ngtc, int *ntrc, int **listc, int *index, unsigned char *maptr, int H, int V, int nclases)
{  int c, i, k, pos=0;
   for(c=0;c<=nclases;c++) if((ntrc[c]<0)||(ntrc[c]>ngtc[c])) ntrc[c]=ngtc[c];
   // construimos maptr
   memset(maptr,0,H*V*sizeof(char));
   for(i=0;i<H*V;i++) if(truth[i]>0) maptr[i]=2;
   for(c=1;c<=nclases;c++) for(i=0;i<ntrc[c];i++)
   {  k=rand()%ngtc[c];
      if(maptr[listc[c][k]]==1) { i--; continue; }
      index[pos]=listc[c][k]; pos++; maptr[listc[c][k]]=1; }
   random_shuffle_bow(index,pos);
}


void mat_mult_btrans_bow(double *A, double *B, double *C, int m, int k, int n)
{ cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,n,k,1,A,k,B,k,0,C,n); }

/* KMEANS*/

double *inv_covariance_matrix_bow(int *datos, int N, int dim)
{  int i, j, *ipiv; double sum, *a, *b;
   a=(double *)malloc((size_t)N*dim*sizeof(double));
   b=(double *)malloc(dim*dim*sizeof(double));
   ipiv=(int *)malloc((dim+1)*sizeof(int));
   if((a==NULL)||(b==NULL)||(ipiv==NULL)) { throw std::bad_alloc() ; }
   for(size_t i=0;i<(size_t)dim*N;i++) a[i]=datos[i];
   cblas_dsyrk(CblasRowMajor,CblasUpper,CblasTrans,dim,N,1.0/N,a,dim,0.0,b,dim);
   // completamos triangular inferior
   for(i=1;i<dim;i++)for(j=0;j<i;j++) b[i*dim+j]=b[j*dim+i];
   // comprobamos que no haya filas con todo ceros
   for(i=0;i<dim;i++)
   {  sum=0; for(j=0;j<dim;j++) sum+=fabs(b[i*dim+j]);
      if(sum<1E-10) b[i*dim+i]=1; }
   // inversa
   if(LAPACKE_dgetrf(LAPACK_COL_MAJOR,dim,dim,b,dim,ipiv)!=0)
      perror("problem with dgetrf");
   // det[k]=0; for(j=0;j<dim;j++) det[k]+=log(fabs(var[k*dim*dim+j*dim+j]));
   // printf("det=%f\n",det[k]);
   if(LAPACKE_dgetri(LAPACK_COL_MAJOR,dim,b,dim,ipiv)!=0)
      perror("problem with dgetri");
   free(a); free(ipiv); return(b);
}


double distance_mahalanobis_bow(double *dat,  double *ivar, int dim)
{  int j; double sum=0;
   double *tmp=(double *)malloc(dim*sizeof(double));
   if(tmp==NULL) { throw std::bad_alloc() ; }
   cblas_dgemv(CblasRowMajor,CblasNoTrans,dim,dim,1.0,ivar,dim,dat,1,0.0,tmp,1);
   for(j=0;j<dim;j++) sum+=tmp[j]*dat[j];
   free(tmp); return(sum);
}


// TIPO: 0=normal, 1=mahalanobis, 2=
inline double compute_distance(double *x, double *ivar, int dim, int tipo, double GAMMA)
{  int i; double dis=0;
   if(tipo==0) for(i=0;i<dim;i++) dis+=x[i]*x[i];
   else if(tipo==1) dis=distance_mahalanobis_bow(x,ivar,dim);
   else if(tipo==2) { for(i=0;i<dim;i++) dis+=x[i]*x[i]; dis=exp(-GAMMA*dis); }
   return(dis);
}


// entradas: data, c; ivar: se computa dentro de la funcion; salidas: result
void kmeans_assignment(int *data, double *c, double *ivar, int N, int dim, int mi, int ma, int K, unsigned char *result, int tipo, double GAMMA)
{  int i, j=0, k, h, ind; double dis, dismin;
   if(tipo==1) ivar=inv_covariance_matrix_bow(data,N,dim);
   else if(tipo==2) GAMMA/=(ma-mi)*(ma-mi)*dim;  // normalizamos

   // para el algoritmo elkan
   double *dat=(double *)malloc(dim*sizeof(double));
   float *dcc=(float *)malloc(K*K*sizeof(float));
   if(dcc==NULL || dat==NULL) { throw std::bad_alloc() ; }

   // iniciamos elkan
   for(i=0;i<K;i++) for(j=i;j<K;j++)
   {  for(k=0;k<dim;k++) dat[k]=c[i*dim+k]-c[j*dim+k];
      dcc[i*K+j]=(float)sqrt(compute_distance(dat,ivar,dim,tipo,GAMMA));
      dcc[j*K+i]=dcc[i*K+j]; }

   // asignacion de datos a clusters
   for(i=0;i<N;i++)
   {  ind=0; dismin=DBL_MAX;
      for(h=0;h<K;h++)
      {  if(dcc[h*K+ind]>=2*dismin) continue;
         for(k=0;k<dim;k++)
          dat[k]=data[(size_t)i*dim+k]-c[h*dim+k];
         dis=sqrt(compute_distance(dat,ivar,dim,tipo,GAMMA));
	       if(dis<dismin) { ind=h; dismin=dis; }
      }
      result[i]=(unsigned char)(ind+1);
   }

   free(dcc); free(dat);
}

//----------------------------------------------------------------------------
/* BOW

datos : vector con los valores espectrales de los pixeles del segmento actual
out : vector de salida
mean: vector con los valores de los centroides (longitud k centroides de B componentes)
N : numero de pixeles del segmento actual
K: numero de centroides
dim : numero de bandas

----------------------------------------------------------------------------*/

void bow_encode(int *datos, double *out, double *mean, int N, int K, int dim)
{
   int i, j, k;
   double center;

   memset(out,0,K*dim*sizeof(double)); //se prepara el vector de salida de tamaño numero de centroides por numero de bandas espectrales
   unsigned char *clase=(unsigned char *)calloc(N,sizeof(char)); //vector de tamaño numero de pixeles del segmento actual
   if(clase==NULL) { printf("Out of memory\n"); exit(-1); }

   kmeans_assignment(datos,mean,NULL,N,dim,0,0,K,clase,0,0); //asignacion de cada uno de los pixeles del segmento actual a alguno de los K centroides
   for(i=0;i<N;i++) clase[i]--; // numeramos en [0:K-1]


   int *mean_vals=(int *)calloc(dim,sizeof(int));
   //(OPTIMIZABLE CON PRAGMA)
   for(i=0;i<K;i++) //para cada centroide
   {  center=0;
      for(j=0;j<N;j++){ //para cada pixel del segmento actual
        if(clase[j]==i) //si el pixel esta asignado al centroide actual
        {
           center += 1; //al final tendra el numero de pixeles del segmento actual asignados al centroide i
           for(k=0;k<dim;k++)
            mean_vals[k] += datos[j*dim+k];
        }
      }

      if(center != 0){
        for(k=0;k<dim;k++)
          out[i*dim+k] = mean_vals[k];
          //out[i*dim+k] = (mean_vals[k]/center);
      }
      else{
        for(k=0;k<dim;k++)
          out[i*dim+k] = -1;
      }
   }


   //normalizacion: da peores resultados que no hacerla
   /*n=0;
   for(k=0;k<dim*K;k++) //para todos los elementos del descriptor de este segmento
   {
     if(out[k] != -1){
       z = out[k];
       n+=z*z;
     }
   }
   n=sqrt(n);
   n=std::max(n,1e-12);
   for(k=0;k<dim*K;k++){
     if(out[k] != -1){
       out[k] /= n; //los divide entre el valor antes calculado
     }

   }*/
   free(clase); //se libera el vector con las asignaciones de cada pixel del segmento al centroide actual
}


// Cuenta el numero de segmentos
// Entrada: label: las etiquetas
// Salidas: tot1:numero de segmentos
// Salidas: mi1, ma1: tamano minimo y maximo de los segmentos
template < typename S >
int *numero_cuencas(S *lab, int H, int V, int *tot1, int *mi1, int *ma1)
{  int u, *count, total=0, ma=0, mi=2147483640;
   count=(int *)calloc(H*V+1,sizeof(int));
   if(count==NULL) { printf("No memory\n"); exit(-1); }
   for(u=0;u<H*V;u++) count[lab[u]]++;
   for(u=0;u<H*V+1;u++) if(count[u]>0) total++;
   //printf("Number of segments=%d\n",total);
   count=(int *)realloc(count,total*sizeof(int));
   for(u=0;u<total;u++)
   {  if(count[u]<mi && count[u]>0) mi=count[u];
      if(count[u]>ma) ma=count[u]; }
   //printf("Segment size, min=%d, max=%d\n",mi,ma);
   //printf("---------------------------------------------------------\n");
   *tot1=total; *ma1=ma; *mi1=mi; return(count);
}


/*
  data : img multiespec

  seg : segmentacion realizada con slic

  B,H,V : numero de bandas, ancho y alto

  model : modelo kmeans

  H1,V1 : parametros de salida (H1 contendra el numero de semgntos)

  K : numero de centroides
*/
int * bow ( unsigned int * data, unsigned int * seg, int B, int H, int V, kmeans_model_t & model, int & H1, int & V1, int & K )
{

  int  i, j, k ;

  //se carga el modelo de kmeans en variables locales
  double *c; //*ivar=NULL,
  //double gamma=0.0;
  //int mi1, ma1, tipo=0;
  c=model.c;//ivar=model.ivar;
  K=model.K;
  //gamma=model.GAMMA;
  //mi1=model.mi;ma1=model.ma;tipo=model.tipo;

  //se cuenta el numero de segmentos y el tam minimo y maximo de cada uno (en cantidad de pixeles que contienen)
  int *data1, HV1, mis, mas ;
  numero_cuencas(seg,H,V,&HV1,&mis,&mas);
  V1=1; //nada
  H1=HV1; //numero de segmentos
   data1= new int [ HV1*K*B ] ( ) ; //descriptores de bow (tantos como segmentos haya, cada uno con k componentes)

   //obtenemos componentes de segmentos
   int nseg=HV1; //numero de segmentos
   int *nclas=(int *)calloc(nseg,sizeof(int)); //vector de tamanho igual al numero de segmentos y que contiene la cantidad de pixeles de cada uno
   int **clas=(int **)malloc(nseg*sizeof(int*)); //vector de tam igual num de segs donde cada elemento es un nuevo vector
   if(nclas==NULL || clas==NULL) { throw std::bad_alloc() ; }

   for(i=0;i<H*V;i++) nclas[seg[i]]++; //se cuenta el numero de pixeles de cada segmento (OPTIMIZABLE)

   for(i=0;i<nseg;i++){ //en cada uno de los nseg elems, se crea un nuevo vector de tamanho el numero de pixeles que contiene esa clase
       clas[i]=(int *)malloc(nclas[i]*sizeof(int));
       if(clas[i]==NULL) { throw std::bad_alloc() ; }
   }

   memset(nclas,0,nseg*sizeof(int)); //se vacia el vector con el num de pixeles de cada seg (POR QUE?OPTIMIZABLE?)

   for(i=0;i<H*V;i++) clas[seg[i]][nclas[seg[i]]++]=i; //posicion de los pixeles de los segmentos dentro de la imagen ordenados
   /*************************************************/


   /*     APLICACION DE BOW A CADA SEGMENTO      */
   //se calcula el numero de pixeles del segmento mas grande (OPTIMIZABLE)
   int ma2=0;
   for(i=0;i<nseg;i++)
      if(nclas[i]>ma2)
        ma2=nclas[i];

   //son vectores auxiliares en el bucle siguiente (OPTIMIZABLES)
   int *datos2=(int *)malloc(ma2*B*sizeof(int)); //vector de tamaño igual al numero de pixeles del segmento mayor y el numero de bandas de este (OPTMIZABLE)
   double *out2=(double *)malloc(ma2*K*B*sizeof(double)); //vector de tamaño igual anterior, pero multiplicado por el numero de centroides
   if(datos2==NULL  || out2==NULL)
   { throw std::bad_alloc() ; }

   //bucle ppal del algoritmo: bow de cada segmento
   for(k=0;k<nseg;k++) //para cada segmento
   {
      if(k%100000==0 || k==(nseg-1)) printf("\tBOW segment %d/%d\n",k,nseg); //cada 100000 segmentos se imprime un mensaje explicativo

      //para cada banda de cada pixel del segmento actual
      for(i=0;i<nclas[k];i++)
        for(j=0;j<B;j++)
          datos2[i*B+j] = data[  clas[k][i]  *B+j]; //asigna los valores espectrales del pixel actual del segmento actual a un vector auxiliar antes creado que es datos2

      bow_encode(datos2,out2,c,nclas[k],K,B); //argumentos: vector con los valores espectrales de los pixeles del segmento actual, vector de salida, vector con los valores de los centroides (longitud k centroides de B componentes)
                                               //            numero de pixeles del segmento actual, numero de centroides, numero de bandas

      //ALGUN OTRO POSTPROCESADO ??
      for(j=0;j<K*B;j++)
        data1[k*K*B+j] = (int)out2[j];
   }


   free(out2); free(datos2); //se liberan los vectores auxiliares anteriores
   /*************************************************/


   return data1;

}
