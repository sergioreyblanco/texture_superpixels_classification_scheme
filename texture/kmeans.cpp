
/**
			  * @file				kmeans.cpp
			  *
				* @author			(C) 2014 Francisco Arguello, Dora B. Heras; Universidade de Santiago de Compostela
				*
			  * @brief      Algoritmo de agrupamiento k-means para imagenes de sensado remoto.
			  */

#include "kmeans.h"


template < typename T >
double *inv_covariance_matrix(T *datos, int N, int dim)
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
   {  printf("ERROR, problema con dgetrf computando varianzas\n"); exit(-1); }
   // det[k]=0; for(j=0;j<dim;j++) det[k]+=log(fabs(var[k*dim*dim+j*dim+j]));
   // printf("det=%f\n",det[k]);
   if(LAPACKE_dgetri(LAPACK_COL_MAJOR,dim,b,dim,ipiv)!=0)
   {  perror("ERROR, problema con dgetri computando varianzas"); exit(-1); }
   free(a); free(ipiv); return(b);
}


double distance_mahalanobis(double *dat,  double *ivar, int dim)
{  int j; double sum=0;
   double *tmp=(double *)malloc(dim*sizeof(double));
   if(tmp==NULL) { throw std::bad_alloc() ; }
   cblas_dgemv(CblasRowMajor,CblasNoTrans,dim,dim,1.0,ivar,dim,dat,1,0.0,tmp,1);
   for(j=0;j<dim;j++) sum+=tmp[j]*dat[j];
   free(tmp); return(sum);
}


// TIPO: 0=normal, 1=mahalanobis, 2=
double compute_distance(double *x, double *ivar, int dim, int tipo, double GAMMA)
{  int i; double dis=0;
   if(tipo==0) for(i=0;i<dim;i++) dis+=x[i]*x[i];
   else if(tipo==1) dis=distance_mahalanobis(x,ivar,dim);
   else if(tipo==2) { for(i=0;i<dim;i++) dis+=x[i]*x[i]; dis=exp(-GAMMA*dis); }
   return(dis);
}


template < typename T >
double *kmeans_hamerley(T *data, double **ivar1, int N, int dim, int mi, int ma, int K, unsigned char *result, int rinit, double eps, int maxit, int tipo, double GAMMA)
{  int a1, r, r1, flag1, flag2, ite=0, i, j=0, k, h, ind=0, pos;
   double dis, dismin, m, errold=0, error=0, dif=0, *ivar=NULL;
   double *c=(double *)malloc(K*dim*sizeof(double));
   int *n=(int*)calloc(K,sizeof(int));
   double *dat=(double *)malloc(dim*sizeof(double));
   if(c==NULL || n==NULL || dat==NULL) { throw std::bad_alloc() ; }
   if(tipo==1) ivar=inv_covariance_matrix(data,N,dim);
   else if(tipo==2) {  GAMMA/=(ma-mi)*(ma-mi)*dim; eps/=(ma-mi); } // normalizamos
   // para el algoritmo hamerly
   double *lb=(double *)malloc(N*sizeof(double));
   double *ub=(double *)malloc(N*sizeof(double));
   double *c1=(double *)malloc(K*dim*sizeof(double));
   double *c2=(double *)malloc(K*dim*sizeof(double));
   double *tmp1=(double *)malloc(K*sizeof(double));
   double *sb=(double *)malloc(K*sizeof(double));
   double *pb=(double *)malloc(K*sizeof(double));
   if(lb==NULL || ub==NULL || c1==NULL || c2==NULL || tmp1==NULL ||
      sb==NULL || pb==NULL) { throw std::bad_alloc() ; }
   // centroides iniciales equiespaciados o random
   if(rinit==0) for(h=0;h<K;h++) for(j=0;j<dim;j++) c[h*dim+j]=data[(size_t)h*(N/K)*dim+j];
   else if(rinit==1) for(h=0;h<K;h++)
   {  pos=rand()%N; for(j=0;j<dim;j++) c[h*dim+j]=data[(size_t)pos*dim+j]; }
   // comprobamos que los centroides no se repitan
   for(h=1;h<K;h++)
   {  flag1=1;
      for(i=0;i<h-1;i++)
      {  flag2=0;
         for(j=0;j<dim;j++) if(c[h*dim+j]!=c[i*dim+j]) { flag2=1; break; }
         if(flag2==0) { flag1=0; break; } }
      if(flag1==0) // hay coincidencia, cambiamos por uno aleatorio
      {  pos=rand()%N;
         for(j=0;j<dim;j++) c[h*dim+j]=data[(size_t)pos*dim+j];
	 h--; }}
   // iniciamos hamerly
   memset(n,0,K*sizeof(int));
   memset(c1,0,K*dim*sizeof(double));
   for(i=0;i<N;i++)
   {  for(h=0;h<K;h++)
      {  for(k=0;k<dim;k++) dat[k]=data[(size_t)i*dim+k]-c[h*dim+k];
         tmp1[h]=sqrt(compute_distance(dat,ivar,dim,tipo,GAMMA)); }
      ind=0; dismin=tmp1[0];
      for(h=0;h<K;h++) if(tmp1[h]<dismin) { ind=h; dismin=tmp1[h]; }
      ub[i]=dismin; result[i]=(unsigned char)ind; tmp1[ind]=DBL_MAX;
      ind=0; dismin=tmp1[0];
      for(h=0;h<K;h++) if(tmp1[h]<dismin) { ind=h; dismin=tmp1[h]; }
      lb[i]=dismin;
      n[result[i]]++;
      for(j=0;j<dim;j++) c1[result[i]*dim+j]+=data[(size_t)i*dim+j]; }
   // comenzamos el algoritmo propiamente dicho
   do { printf("\tkmeans hamerley, iter %d, dif=%f\n",ite,dif);
      errold=error; error=0;
      for(i=0;i<K;i++)
      {  ind=0; dismin=DBL_MAX;
         for(j=0;j<K;j++)
	 {  if(i==j) continue;
	    for(k=0;k<dim;k++) dat[k]=c[i*dim+k]-c[j*dim+k];
	    dis=sqrt(compute_distance(dat,ivar,dim,tipo,GAMMA));
	    if(dis<dismin) dismin=dis; }
	 sb[i]=dismin; }
      for(i=0;i<N;i++)
      {  m=(((sb[result[i]]/2)>(lb[i]))?(sb[result[i]]/2):(lb[i]));
         if(ub[i]>m)
	 {  for(k=0;k<dim;k++) dat[k]=data[(size_t)i*dim+k]-c[result[i]*dim+k];
	    ub[i]=sqrt(compute_distance(dat,ivar,dim,tipo,GAMMA));
	    if(ub[i]>m)
	    {  a1=result[i];
	       // igual que el inicio
	       for(h=0;h<K;h++)
	       {  for(k=0;k<dim;k++) dat[k]=data[(size_t)i*dim+k]-c[h*dim+k];
                  tmp1[h]=sqrt(compute_distance(dat,ivar,dim,tipo,GAMMA)); }
               ind=0; dismin=tmp1[0];
               for(h=0;h<K;h++) if(tmp1[h]<dismin) { ind=h; dismin=tmp1[h]; }
               ub[i]=dismin; result[i]=(unsigned char)ind; tmp1[ind]=DBL_MAX;
               ind=0; dismin=tmp1[0];
               for(h=0;h<K;h++) if(tmp1[h]<dismin) { ind=h; dismin=tmp1[h]; }
               lb[i]=dismin;
	       if(a1!=result[i])
	       {  n[a1]--; n[result[i]]++;
	          for(j=0;j<dim;j++)
		  {  c1[a1*dim+j]-=data[(size_t)i*dim+j]; c1[result[i]*dim+j]+=data[(size_t)i*dim+j]; } } } } }
      memcpy(c2,c,K*dim*sizeof(double));
      for(h=0;h<K;h++)
      {  if(n[h]!=0) for(j=0;j<dim;j++) c[h*dim+j]=c1[h*dim+j]/n[h];
         else { pos=rand()%N; for(j=0;j<dim;j++) c[h*dim+j]=data[(size_t)pos*dim+j]; } }
      for(h=0;h<K;h++)
      {  for(k=0;k<dim;k++) dat[k]=c2[h*dim+k]-c[h*dim+k];
         pb[h]=sqrt(compute_distance(dat,ivar,dim,tipo,GAMMA)); }
      for(h=0;h<K;h++) error+=pb[h];
      r=0; dis=pb[0]; for(h=0;h<K;h++) { if(pb[h]>dis) { dis=pb[h]; r=h; } }
      r1=0; dis=DBL_MIN; for(h=0;h<K;h++) { if(h!=r && pb[h]>dis) { dis=pb[h]; r1=h; } }
      for(i=0;i<N;i++)
      {  ub[i]+=pb[result[i]];
         if(r==result[i]) lb[i]-=pb[r1]; else lb[i]-=pb[r]; }
      error/=(K*dim); ite++;
   } while(((dif=fabs(error-errold))>eps)&&(ite<maxit));

   for(i=0;i<N;i++) result[i]++;
   free(n); free(dat); free(lb); free(ub); free(c1); free(c2); free(tmp1);
   free(sb); free(pb); *ivar1=ivar; return(c);
}


void kmeans_save_model(double *c, double *ivar, int B, int K, double GAMMA, int mi, int ma, int tipo, const char * filename)
{  FILE *fp; int code=401;
   fp=fopen(filename,"w");
   if(fp==NULL) { throw -1 ; }
   if(tipo==0) fprintf(fp,"KMEANS\nB=%d\nK=%d\nmin=%d\nmax=%d\n",B,K,mi,ma);
   else if(tipo==1) fprintf(fp,"KMH\nB=%d\nK=%d\nmin=%d\nmax=%d\n",B,K,mi,ma);
   else if(tipo==2) fprintf(fp,"KKM\nB=%d\nK=%d\ngamma=%f\nmin=%d\nmax=%d\n",B,K,GAMMA,mi,ma);
   fwrite(&code,4,1,fp); // code
   fwrite(c,sizeof(double),K*B,fp);
   if(tipo==1) fwrite(ivar,sizeof(double),B*B,fp);
   fclose(fp);
}


void kmeans_save_model ( kmeans_model_t const & model, std::string const & save_path )
{
        kmeans_save_model(model.c, model.ivar, model.B, model.K, model.GAMMA, model.mi, model.ma, model.tipo, save_path.c_str ( ) ) ;
}


int kmeans_load_model(double **c1, double **ivar1, int B1, int *K1, double *GAMMA1, int *mi1, int *ma1, int tipo, int flag, const char *filename)
{  FILE *fp; int B, K, code, mi, ma; char texto[20]; size_t s1;
   double *c=NULL, *ivar=NULL, GAMMA=0;
   fp=fopen(filename,"r");
   if(fp==NULL) { throw -1 ; }
   s1=fscanf(fp,"%s\n",texto);
   if(tipo==0 && strncmp(texto,"KMEANS",7)!=0)
   { printf("ERROR: Modelo incorrecto %s\n",texto); exit(-1); }
   else if(tipo==1 && strncmp(texto,"KMH",4)!=0)
   { printf("ERROR: Modelo incorrecto %s\n",texto); exit(-1); }
   else if(tipo==2 && strncmp(texto,"KKM",4)!=0)
   { printf("ERROR: Modelo incorrecto %s\n",texto); exit(-1); }
   if(tipo==0 || tipo==1) s1=fscanf(fp,"B=%d\nK=%d\nmin=%d\nmax=%d\n",&B,&K,&mi,&ma);
   else if(tipo==2) s1=fscanf(fp,"B=%d\nK=%d\ngamma=%f\nmin=%d\nmax=%d\n",&B,&K,(float*)&GAMMA,&mi,&ma);
   s1=fread(&code,4,1,fp); // code
   if(B!=B1) { printf("ERROR: mal parametro, B=%d, %d\n",B,B1); exit(-1); }
   if(flag==1) { *K1=K; *mi1=mi; *ma1=ma; if(tipo==2) *GAMMA1=GAMMA; }
      printf("* Modelo leido %s, B=%d, K=%d, GAMMA=%f\n",texto,B,K,GAMMA);
      c=(double *)malloc(K*B*sizeof(double));
      if(c==NULL) { throw std::bad_alloc() ; }
      s1=fread(c,sizeof(double),K*B,fp);
      *c1=c;
      if(tipo==1)
      {  ivar=(double *)malloc(B*B*sizeof(double));
         if(ivar==NULL) { throw std::bad_alloc() ; }
         s1=fread(ivar,sizeof(double),B*B,fp);
	 *ivar1=ivar; }
   if(s1<=0) { printf("ERROR: Error de lectura\n"); exit(-1); }
   fclose(fp); return(1);
}


kmeans_model_t kmeans_load_model( std::string const & load_path, int const & B, int const & tipo )
{
    kmeans_model_t  model ;
    kmeans_load_model(&model.c, &model.ivar, B , &model.K, &model.GAMMA, &model.mi, &model.ma, tipo, 1, load_path.c_str ( ) ) ;
    return model ;
}

void destroy_kmeans_model ( kmeans_model_t & model )
{
    free ( model.c ) ;
    if ( model.ivar != NULL ) {
        free ( model.ivar ) ;
    }
}


void kmeans(unsigned int *datos, int N, int dim, struct kmeans_parameter_t params,  struct kmeans_model_t* model )
{
    int K = params.K ;
    int rinit = params.RINIT ;
    double eps = params.EPS ;
    int maxit = params.MAXIT ;
    int tipo = params.TIPO ;
    int i, mi, ma; double *ivar=NULL, *c=NULL, gamma=1;

    if ( K <= 0 ) {
        throw std::invalid_argument (
                "K-means error. The number of centroids K to be computed by k-means is less or equal than zero. It must be a positive number." ) ;
    }

    if ( rinit != 1 && rinit != 0 ) {
        throw std::invalid_argument ( "K-means error. The initialization flag is invalid. Allowed values: '0' or '1'." ) ;
    }

    if ( maxit <= 0 ) {
        throw std::invalid_argument ( "K-means error. The maximum number of iterations must be a positive integer." ) ;
    }

    if ( tipo != 0 && tipo != 1 && tipo != 2 ) {
        throw std::invalid_argument (
                "K-means error. The the distance flag is invalid. Allowed values: '0', '1' or '2'." ) ;
    }

   printf("\t* Kmeans, clusters=%d, inicio=%d, tolerancia=%f, maxit=%d, tipo=%d\n",
      K,rinit,eps,maxit,tipo);
   mi=datos[0]; ma=datos[0];
   for(i=0;i<N;i++)
   { if(datos[i]<(unsigned int)mi) mi=(int)datos[i]; if(datos[i]>(unsigned int)ma) ma=(int)datos[i]; }
   unsigned char *result= new unsigned char [ N ] ; //(unsigned char *)malloc(N*sizeof(char));

   c=kmeans_hamerley(datos,&ivar,N,dim,mi,ma,K,result,rinit,eps,maxit,tipo,gamma);

   struct kmeans_model_t p { c,ivar,dim,K,gamma,mi,ma,tipo } ;
   *model = p ;
   /* DO NOT FREE, struct kmeans_model_t does NOT hold a cpy assig. !! */
   //free(c); free(ivar);

   delete [] result ;
}
