
/**
			  * @file				slic.cpp
			  *
				* @author			-
				*
			  * @brief      SLIC algorithm for segmented image computation.
			  */

#include "slic.h"

  //----------------------------------------------------------------------------
  // OPERACIONES CON SEGMENTOS/OBJETOS
  //----------------------------------------------------------------------------
  template <typename T>
  T abs_slic(T value)
  {
    if(value < 0){
      return value*-1;
    }
    else{
      return value;
    }
  }


  template < typename T >
  int relabel(T *img, int *lab, int u, int v)
  {  if((img[u]==img[v])&&(lab[u]>lab[v])) { lab[u]=lab[v]; return(1); }
     else return(0);
  }


  // hace que todos los segmentos esten conectados, dividiendolos
  template < typename T >
  void ajusta_conectividad(T *img, int H, int V, int CONN)
  {  int u, v, *lab, flag;
     printf("\t* Reparamos conectividad\n");
     lab=(int *)calloc(H*V,sizeof(int));
     if(lab==NULL) { perror("mem"); exit(-1); }
     for(u=0;u<H*V;u++) lab[u]=u;
     flag=1;
     while(flag)
     {  flag=0;
        for(u=0;u<H*V;u++)
        { if(u>=H)     { v=u-H; flag += relabel(img,lab,u,v); }
          if(u<H*V-H)  { v=u+H; flag += relabel(img,lab,u,v); }
          if(u%H!=0)   { v=u-1; flag += relabel(img,lab,u,v); }
          if(u%H!=H-1) { v=u+1; flag += relabel(img,lab,u,v); }
          if(CONN==8)
          {  if((u>=H)&&(u%H!=0))      { v=u-H-1; flag += relabel(img,lab,u,v); }
             if((u>=H)&&(u%H!=H-1))    { v=u-H+1; flag += relabel(img,lab,u,v); }
             if((u<H*V-H)&&(u%H!=0))   { v=u+H-1; flag += relabel(img,lab,u,v); }
             if((u<H*V-H)&&(u%H!=H-1)) { v=u+H+1; flag += relabel(img,lab,u,v); } } }}
     memcpy(img,lab,(size_t)H*V*sizeof(int));
     free(lab);
  }


  // renumera las etiquetas para que sean consecutivas
  template < typename T >
  int adjust_segments(T *img, int H, int V)
  {  int u, tot=0;
     int *lab=(int *)malloc((size_t)H*V*sizeof(int));
     int *ind=(int *)malloc((H*V+1)*sizeof(int));
     if((lab==NULL)||(ind==NULL)) { printf("No memory\n"); exit(-1); }
     for(u=0;u<H*V+1;u++) ind[u]=-1;
     for(u=0;u<H*V;u++) if(ind[img[u]]==-1) ind[img[u]]=tot++;
     for(u=0;u<H*V;u++) lab[u]=ind[img[u]];
     memcpy(img,lab,(size_t)H*V*sizeof(int));
     free(ind); free(lab);
     return(tot);
  }


  // construye la tabla neig con las posiciones de los vecinos de cada pixel
  // necesita CONN posiciones
  template < typename T >
  int obtain_neig(T *img, int H, int V, int *neig, int u, int CONN)
  {  int k=0;
     if(u%H>0)   neig[k++]=img[u-1];
     if(u%H<H-1) neig[k++]=img[u+1];
     if(u>=H)    neig[k++]=img[u-H];
     if(u<V*H-H) neig[k++]=img[u+H];
     else if(CONN==8)
     {  if((u>=H)&&(u%H>0))      neig[k++]=img[u-H-1];
        if((u>=H)&&(u%H<H-1))    neig[k++]=img[u-H+1];
        if((u<V*H-H)&&(u%H>0))   neig[k++]=img[u+H-1];
        if((u<V*H-H)&&(u%H<H-1)) neig[k++]=img[u+H+1]; }
     return(k);
  }


  double distancia_centros(double *c, int B, int p1, int p2)
  {  int i; double dis=0;
     for(i=0;i<B;i++) dis+=(c[p1*B+i]-c[p2*B+i])*(c[p1*B+i]-c[p2*B+i]);
     return(sqrt(dis));
  }


  template < typename T >
  int adjust_size_segments_distance(int *img, T *datos, int H, int V, int B, int siz, int CONN)
  {  int u, i, k, ite, pos, tot=0, maxtot=0, *count, *ind, neig[8], nvec, num;
     struct obj *ob1=NULL; double *c, mi, dis;
     ajusta_conectividad(img,H,V,CONN);
     num=adjust_segments(img,H,V);
     count=(int *)calloc(num,sizeof(int));
     ind=(int *)malloc(num*sizeof(int));
     c=(double *)calloc(num*B,sizeof(double));
     if((count==NULL)||(ind==NULL)||(c==NULL)) { printf("No hay memoria\n"); exit(-1); }
     for(ite=0;ite<100;ite++)
     {  tot=0;
        memset(count,0,num*sizeof(int));
        for(u=0;u<H*V;u++) count[img[u]]++;
        for(u=0;u<num;u++) if(count[u]<siz) ind[u]=tot++;
        printf("\tAjuste segmentos: %d (%d con menos de %d pixels)\n",num,tot,siz);
        if(tot>maxtot) maxtot=tot;
        if(tot==0) break;
        // calculamos centros
        memset(c,0,num*B*sizeof(double));
        for(u=0;u<H*V;u++) for(i=0;i<B;i++) c[img[u]*B+i]+=datos[u*B+i];
        for(u=0;u<num;u++) if(count[u]!=0) for(i=0;i<B;i++) c[u*B+i]/=count[u];
        //if(IMPRIME) { for(i=0;i<num;i++) { printf("%d ",i);
        //    if(count[i]<siz) printf("(%d), ",count[i]); } printf("\n"); }
        // para cada segmento menor al tamano especificado hacemos una tabla de vecinos
        if(ob1==NULL)
        {  ob1=(struct obj*)calloc(tot,sizeof(struct obj));
           if(ob1==NULL) { printf("No hay memoria\n"); exit(-1); }
  	 for(i=0;i<tot;i++)
  	 {  ob1[i].siz=OBS;
  	    ob1[i].vec=(int *)malloc(OBS*sizeof(int));
  	    if(ob1[i].vec==NULL) { printf("No hay memoria\n"); exit(-1); } } }
        else for(i=0;i<tot;i++) ob1[i].n=0;
        for(u=0;u<H*V;u++) if(count[img[u]]<siz)
        {  pos=ind[img[u]];
           // si la posicion en la tabla esta vacia la iniciamos
           if(ob1[pos].n==0) ob1[pos].lab=img[u];
           nvec=obtain_neig(img,H,V,neig,u,CONN);
           for(k=0;k<nvec;k++)
           {  if(img[u]==neig[k]) continue;
  	    // miramos si el vecino ya lo tenemos de antes o es nuevo
              for(i=0;i<ob1[pos].n;i++) if(ob1[pos].vec[i]==neig[k]) break;
              if(i==ob1[pos].n)
  	    {  if(ob1[pos].n==ob1[pos].siz)
  	       {  // printf("  ampliamos vecinos a %d\n",ob1[pos].siz+OBS);
  	          ob1[pos].vec=(int *)realloc(ob1[pos].vec,(OBS+ob1[pos].siz)*sizeof(int));
  	          ob1[pos].siz+=OBS; }
  	       ob1[pos].vec[i]=neig[k]; ob1[pos].n++; } } }
        // completada la tabla, buscamos el segmento vecino de menor distancia
        for(u=0;u<tot;u++)
        {  pos=0; mi=DBL_MAX;
           for(i=0;i<ob1[u].n;i++)
           {  dis=distancia_centros(c,B,ob1[u].lab,ob1[u].vec[i]);
  	    if(dis<mi) { mi=dis; pos=i; } }
  	 ob1[u].pvec=ob1[u].vec[pos];
           // actualizamos las tablas porque vamos a seguir mirando segmentos
  	 // es muy lento, lo quitamos, es preferible tener un mayor numero de iteraciones
  	 // for(k=0;k<u;k++) if(ob1[k].pvec==ob1[u].lab) ob1[k].pvec=ob1[u].pvec;
           for(k=0;k<ob1[u].n;k++) if(count[ob1[u].vec[k]]<siz)
           {  pos=ind[ob1[u].vec[k]];
              for(i=0;i<ob1[pos].n;i++)
  	       if(ob1[pos].vec[i]==ob1[u].lab) ob1[pos].vec[i]=ob1[u].pvec; } }
        // hacemos la fusion del segmenos con el que estamos trabajando
        for(u=0;u<H*V;u++) if(count[img[u]]<siz) img[u]=ob1[ind[img[u]]].pvec;
        num=adjust_segments(img,H,V); }
     printf("\tAjuste realizado en %d iteraciones\n",ite);
     free(count); free(ind); for(i=0;i<maxtot;i++) free(ob1[i].vec); free(ob1); free(c);
     return(num);
  }


  int numero_cuencas(int *lab, int H, int V)
  {  int u, *count, total=0;
     count=(int *)calloc(H*V+1,sizeof(int));
     for(u=0;u<H*V;u++) count[lab[u]]++;
     for(u=0;u<H*V+1;u++) if(count[u]>0) total++;
     printf("\t* Segmentos obtenidos: %d\n",total);
     return(total);
  }


  template < typename T >
  void test_conectividad(T* img, int H, int V, int CONN)
  {  int u, v, *lab, flag;
     lab=(int *)calloc(H*V,sizeof(int));
     if(lab==NULL) { perror("mem"); exit(-1); }
     for(u=0;u<H*V;u++) lab[u]=u;
     flag=1;
     while(flag)
     {  flag=0;
        for(u=0;u<H*V;u++)
        { if(u>=H)     { v=u-H; flag += relabel(img,lab,u,v); }
          if(u<H*V-H)  { v=u+H; flag += relabel(img,lab,u,v); }
          if(u%H!=0)   { v=u-1; flag += relabel(img,lab,u,v); }
          if(u%H!=H-1) { v=u+1; flag += relabel(img,lab,u,v); }
          if(CONN==8)
          {  if((u>=H)&&(u%H!=0))      { v=u-H-1; flag += relabel(img,lab,u,v); }
             if((u>=H)&&(u%H!=H-1))    { v=u-H+1; flag += relabel(img,lab,u,v); }
             if((u<H*V-H)&&(u%H!=0))   { v=u+H-1; flag += relabel(img,lab,u,v); }
             if((u<H*V-H)&&(u%H!=H-1)) { v=u+H+1; flag += relabel(img,lab,u,v); } } }}
     numero_cuencas(lab,H,V);
  }


  // ---------------------------------------------------------------
  // SLIC propiamente dicho
  // ---------------------------------------------------------------

  template < typename T >
  double grad(T *img, int x, int y, int H, int V, int dim)
  {  int i, j, k, posy, posx; double out=0, sum;
     for(i=-1;i<2;i++) for(j=-1;j<2;j++)
     {  posy=i+y; posx=j+x;
        if((posy<0)||(posy>=V)||(posx<0)||(posx>=H)) continue;
        sum=0;
        for(k=0;k<dim;k++) sum+=abs_slic(img[(posy*H+posx)*dim+k]-img[(y*H+x)*dim+k]);
        if(sum>out) out=sum; }
     return(out);
  }


  template < typename T >
  void grad_mov(T *img, int *x, int *y, int H, int V, int dim)
  {  int i, j, posy, posx, indy, indx; double out, tmp;
     indy=*y; indx=*x; out=grad(img,indx,indy,H,V,dim);
     for(i=-1;i<2;i++) for(j=-1;j<2;j++)
     {  posy=i+*y; posx=j+*x;
        if((posy<0)||(posy>=V)||(posx<0)||(posx>=H)) continue;
        tmp=grad(img,posx,posy,H,V,dim);
        if(tmp<out) { out=tmp; indy=posy; indx=posx; } }
     // printf("%d,%d, %d,%d, %ld\n",*x,indx,*y,indy,out);
     *y=indy; *x=indx;
  }


  template < typename T >
  double distance(T *img, double *c, int x, int y, int H, int V, int dim, double scal)
  {  double sum; int i;
     sum=scal*((c[1]-y)*(c[1]-y)+(c[0]-x)*(c[0]-x));
     for(i=0;i<dim;i++) sum+=(c[i+2]-img[(y*H+x)*dim+i])
                            *(c[i+2]-img[(y*H+x)*dim+i]);
     return(sqrt(sum));
  }


  int * slic (unsigned int *img, int H, int V, int dim, struct slic_parameter_t params, int* number_segments)
  {
      int S = params.S;
      int m = params.m ;
      int minsize = params.minsize ;
      int CONN = params.CONN ;
      double THRES = params.threshold ;
      int i, j, k, Kh, Kv, K, x, y, ite=0, mi, ma;
     double D, dismin=INF, dif, error=INF, errold, scal;
     Kh=(int)round(1.0*H/S-0.1); Kv=(int)round(1.0*V/S-0.1); K=Kh*Kv;
     double *c=(double *)malloc(K*(dim+2)*sizeof(double));
     double *dis=(double *)malloc((size_t)H*V*sizeof(double));
     int *lab=(int *)malloc((size_t)H*V*sizeof(int));
     int *n=(int *)malloc((size_t)H*V*sizeof(int));
     int *sum=(int *)calloc(K,sizeof(int));
     if((c==NULL)||(lab==NULL)||(dis==NULL)||(n==NULL)||(sum==NULL))
     { printf("No hay memoria\n"); exit(0); }
     printf("\t* SLIC, long=%d, regul=%d, tam_min=%d, conn=%d, tol=%f\n",
        S,m,minsize,CONN,THRES);
     assert(2*S<H && 2*S<V);
     for(i=0;i<H*V;i++) dis[i]=-1;
     mi=img[0]; ma=img[0];
     for(size_t i=0;i<(size_t)H*V*dim;i++) { if(img[i]<(unsigned int)mi) mi=(int)img[i]; if(img[i]>(unsigned int)ma) ma=(int)img[i]; }
     scal=1.0*m*m*dim*(ma-mi)*(ma-mi)/(S*S*256*256);
     for(i=0;i<Kv;i++) for(j=0;j<Kh;j++)
     {  x=S/2+j*S; y=S/2+i*S;
        if(x>H || y>V) printf("\n\nERROR\n");
        grad_mov(img,&x,&y,H,V,dim);
        c[(i*Kh+j)*(dim+2)+0]=x;
        c[(i*Kh+j)*(dim+2)+1]=y;
        for(k=0;k<dim;k++) c[(i*Kh+j)*(dim+2)+k+2]=img[k]; }
     do
     {  printf("\tSLIC, iteracion %d, dif=%f\n",ite,dif);
        errold=error; error=0;
        for(i=0;i<H*V;i++) dis[i]=INF;
        for(k=0;k<Kv*Kh;k++)
        {  y=(int)c[k*(dim+2)+1]; x=(int)c[k*(dim+2)+0];
           for(i=-S;i<=S;i++) for(j=-S;j<=S;j++)
           {  if((y+i<0)||(y+i>=V)||(x+j<0)||(x+j>=H)) continue;
  	    D=distance(img,c+k*(dim+2),x+j,y+i,H,V,dim,scal);
  	    if(D<dis[(y+i)*H+(x+j)])
  	    {  dismin=D; dis[(y+i)*H+(x+j)]=D; lab[(y+i)*H+(x+j)]=k; } }
  	 error+=dismin; }
        ite++; error/=((double)H*V*(dim+2));
        // relabellizacion de centroides
        memset(n,0,(size_t)H*V*sizeof(int));
        for(i=0;i<H*V;i++) n[lab[i]]++;
        memset(c,0,K*(dim+2)*sizeof(double));
        for(i=0;i<H*V;i++)
        {  if(lab[i]<0 || lab[i]>K) printf("\n\nERROR %d, %d\n\n",i,lab[i]);
           c[lab[i]*(dim+2)+1]+=i/H; c[lab[i]*(dim+2)+0]+=i%H;
           for(j=0;j<dim;j++) c[lab[i]*(dim+2)+j+2]+=img[i*dim+j]; }
        for(k=0;k<H*V;k++) if(n[k]!=0) for(j=0;j<dim+2;j++) c[k*(dim+2)+j]/=n[k];
     } while((dif=abs_slic(error-errold))>THRES && ite<100 );

     adjust_size_segments_distance(lab,img,H,V,dim,minsize,CONN);
     free(c); free(dis); free(n); free(sum);

     (*number_segments) = numero_cuencas(lab, H, V);

     return(lab);
  }
