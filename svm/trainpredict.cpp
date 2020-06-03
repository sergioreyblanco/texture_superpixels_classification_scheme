
/**
			  * @file				trainpredict.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Training and testing phases with the SVM engine.
			  */

#include "trainpredict.h"




/******************************************** TRAIN **************************************/

char* readline_train(FILE *input, char *line, int max_line_len)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}


void load_problem_txt(const char *filename, struct svm_node * X_matrix, struct svm_parameter param, struct svm_problem* prob, char* error)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		sprintf(error, "can't open input file %s", filename);
		print_error((char*)error);
		exit(1);
	}

	prob->l = 0;
	elements = 0;

	int max_line_len = 1024;
	char* line = (char*)malloc(sizeof(char)*max_line_len);
	while(readline_train(fp, line, max_line_len)!=NULL)
	{
		char *p = strtok(line," \t"); // label


		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}

		++elements;
		++prob->l;
	}
	rewind(fp);

	prob->y = (double*)malloc(sizeof(double)*prob->l);
	prob->x = (struct svm_node **)malloc(sizeof(struct svm_node *)*prob->l);
	X_matrix = (struct svm_node *)malloc(sizeof(struct svm_node )*elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob->l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline_train(fp, line, max_line_len);
		prob->x[i] = &X_matrix[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob->y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			X_matrix[j].index = (int) strtol(idx,&endptr,10);

			if(endptr == idx || errno != 0 || *endptr != '\0' || X_matrix[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = X_matrix[j].index;

			errno = 0;
			X_matrix[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		X_matrix[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob->l;i++)
		{
			if (prob->x[i][0].index != 0)
			{
				sprintf(error, "Wrong input format: first column must be 0:sample_serial_number");
				print_error((char*)error);
				exit(1);
			}
			if ((int)prob->x[i][0].value <= 0 || (int)prob->x[i][0].value > max_index)
			{
				sprintf(error, "Wrong input format: sample_serial_number out of range");
				print_error((char*)error);
				exit(1);
			}
		}

	fclose(fp);
	free(line);
}


short load_problem_hsi(image_struct *train_image, reference_data_struct *gt_train_image, int instances, struct svm_problem* prob, struct svm_node *X_matrix)
{

	//SVM problem
	prob->l = instances;
	prob->y = (double*)malloc(sizeof(double)*prob->l);
	prob->x = (struct svm_node **)malloc(sizeof(struct svm_node*)*prob->l);

	//loading loop
	X_matrix = (struct svm_node *)malloc(sizeof(struct svm_node)*instances*(get_image_bands(train_image)+1));
	int j=0;
	for(int i=0;i<prob->l;i++){
		prob->x[i] = &X_matrix[j];

		prob->y[i] = get_reference_data_data(gt_train_image)[i];

		for(unsigned int b=0;b<get_image_bands(train_image);b++) {
			X_matrix[j].index = b;
			X_matrix[j].value = get_image_data(train_image)[i*get_image_bands(train_image)+b];

			++j;
		}

		X_matrix[j++].index = -1;
	}


	return EXIT_SUCCESS;
}


short load_problem_texture(texture_struct *descriptors_train, struct svm_problem* prob, struct svm_node *X_matrix)
{

	/*for(int i=0; i<get_descriptors_number_descriptors(descriptors_train);i++){
		printf("\n\n%d\n", get_descriptors_labels(descriptors_train)[i]);

		for(int b=0;b<get_descriptors_dim_descriptors(descriptors_train);b++) {
			printf("%d  ", get_descriptors_data(descriptors_train)[i*get_descriptors_dim_descriptors(descriptors_train)+b]);
		}
	}*/

	//SVM problem
	prob->l = get_descriptors_number_descriptors(descriptors_train);
	prob->y = (double*)malloc(sizeof(double)*prob->l);
	prob->x = (struct svm_node **)malloc(sizeof(struct svm_node*)*prob->l);

	//loading loop

	X_matrix = (struct svm_node *)malloc(sizeof(struct svm_node)*get_descriptors_number_descriptors(descriptors_train)*(get_descriptors_dim_descriptors(descriptors_train)+1));
	int j=0;
	for(int i=0;i<get_descriptors_number_descriptors(descriptors_train);i++){
		prob->x[i] = &X_matrix[j];

		prob->y[i] = (float)get_descriptors_labels(descriptors_train)[i];

		for(int b=0;b<get_descriptors_dim_descriptors(descriptors_train);b++) {
			X_matrix[j].index = b;
			X_matrix[j].value = get_descriptors_data(descriptors_train)[i*get_descriptors_dim_descriptors(descriptors_train)+b];

			++j;
		}

		X_matrix[j++].index = -1;
	}

	return EXIT_SUCCESS;
}


void do_cross_validation(int nr_fold, struct svm_parameter param, struct svm_problem prob)
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double* target = (double*)malloc(sizeof(double)*prob.l);

	svm_cross_validation(&prob,&param,nr_fold,target); //todo: sustituir por svm_train
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}
	free(target);
}








/******************************************** PREDICT **************************************/

char* readline_predict(FILE *input, char *line, int max_line_len)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}


void predict_txt(FILE *input, FILE *output, svm_model *svm_model, struct svm_parameter param)
{
	int max_nr_attr = 64;
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	struct svm_node *x;

	int svm_type=svm_get_svm_type(svm_model);
	int nr_class=svm_get_nr_class(svm_model);
	double *prob_estimates=NULL;
	int j;

	if(param.probability)
	{
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
			printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(svm_model));
		else
		{
			int *labels=(int *) malloc(nr_class*sizeof(int));
			svm_get_labels(svm_model,labels);
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"labels");
			for(j=0;j<nr_class;j++)
				fprintf(output," %d",labels[j]);
			fprintf(output,"\n");
			free(labels);
		}
	}



	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	int max_line_len = 1024;
	char* line = (char *)malloc(max_line_len*sizeof(char));
	while(readline_predict(input, line, max_line_len) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		max_nr_attr = 64;
		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		x[i].index = -1;





		if (param.probability && (svm_type==C_SVC || svm_type==NU_SVC))
		{
			predict_label = svm_predict_probability(svm_model,x,prob_estimates);
			fprintf(output,"%g",predict_label);
			for(j=0;j<nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else
		{
			predict_label = svm_predict(svm_model,x);
			fprintf(output,"%.17g\n",predict_label);
		}

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		printf("Mean squared error = %g (regression)\n",error/total);
		printf("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
		printf("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);
	if(param.probability)
		free(prob_estimates);

	free(line);
}


/**			PRIVATE FUNTCION
				 * @brief      -
				 *
				 * @param      -
         * @param      -
         * @param      -
         * @param      -
				 * @param      -
				 * @param      -
				 *
				 * @return     -
				 */
void mat_mult_btrans(double *A, double *B, double *C, int m, int k, int n)
{
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,n,k,1,A,k,B,k,0,C,n);
}


/**			PRIVATE FUNTCION
				 * @brief      Makes a prediction for a block of pixels
				 *
				 * @param      model  SVM model created in the training phase
         * @param      sv Suppport vectors
         * @param      x Block of pixels
				 * @param      result Block of pixels
				 * @param      BLOCK Size of the block of pixels
				 * @param      case Type of SVM kernel
				 * @param      B Number of bands of the pixels
				 *
				 * @return     -
				 */
void svm_predict_block(const svm_model *model, double *sv, double *x, unsigned char *result, int BLOCK, int caso, int B)
{
	 int i, j, k, p, d, si, sj, ci, cj, vote_max_idx, nr_class=model->nr_class, L=model->l;
	 double sum, GAMMA=(model->param.gamma), *coef1, *coef2, *x2=NULL, *s2=NULL;
   double *kvalue=(double *)malloc(BLOCK*L*sizeof(double));
   int *start=(int *)malloc(nr_class*sizeof(int));
   int *vote=(int *)malloc(nr_class*sizeof(int));
   if(kvalue==NULL || start==NULL || vote==NULL)
   {  printf("Out of memory\n"); exit(-1); }
   start[0]=0; for(i=1;i<nr_class;i++) start[i]=start[i-1]+model->nSV[i-1];
   // calculamos el kernel
   if(caso==1 || caso==2)
   {  x2=(double *)calloc(BLOCK,sizeof(double));
      s2=(double *)calloc(L,sizeof(double));
      if(x2==NULL || s2==NULL) { printf("out of memory\n"); exit(-1); }
      for(d=0;d<BLOCK;d++) for(j=0;j<B;j++) x2[d]+=x[d*B+j]*x[d*B+j];
      for(i=0;i<L;i++) for(j=0;j<B;j++) s2[i]+=sv[i*B+j]*sv[i*B+j]; }
   mat_mult_btrans(x,sv,kvalue,BLOCK,B,L);
   // 1=RBF, 2=LAP, 3=LIN, 4=POLY, 5=SAM, 6=SIG
   if(caso==1)
   {  for(d=0;d<BLOCK;d++) for(i=0;i<L;i++) kvalue[d*L+i]=x2[d]+s2[i]-2*kvalue[d*L+i];
      for(i=0;i<BLOCK*L;i++) kvalue[i]=exp(-GAMMA*kvalue[i]); }
   else if(caso==2)
   {  for(d=0;d<BLOCK;d++) for(i=0;i<L;i++) kvalue[d*L+i]=x2[d]+s2[i]-2*kvalue[d*L+i];
      for(i=0;i<BLOCK*L;i++) kvalue[i]=exp(-sqrt(GAMMA*kvalue[i])); }
   else if(caso==3) {;} // devolvemos el producto sin mas
   else if(caso==4)
   {  for(i=0;i<BLOCK*L;i++) kvalue[i]=1+kvalue[i]*GAMMA;
      for(i=0;i<BLOCK*L;i++) kvalue[i]=kvalue[i]*kvalue[i]; }
   else if(caso==5)
   {  for(i=0;i<BLOCK*L;i++) kvalue[i]=acos(kvalue[i]*GAMMA);
      for(i=0;i<BLOCK*L;i++) kvalue[i]=exp(-kvalue[i]*kvalue[i]); }
   else if(caso==6)
   {  for(i=0;i<BLOCK*L;i++) kvalue[i]=tanh(kvalue[i]*GAMMA); }
   else printf("No tal kernel\n");
   free(x2); free(s2);
   // resto de las operaciones
   for(d=0;d<BLOCK;d++)
   {  p=0; memset(vote,0,nr_class*sizeof(int));
      for(i=0;i<nr_class;i++) for(j=i+1;j<nr_class;j++)
      {  sum=-model->rho[p]; si=start[i]; sj=start[j]; ci=model->nSV[i]; cj=model->nSV[j];
         coef1=model->sv_coef[j-1]; coef2=model->sv_coef[i];
         for(k=0;k<ci;k++) sum+=coef1[si+k]*kvalue[d*L+si+k];
         for(k=0;k<cj;k++) sum+=coef2[sj+k]*kvalue[d*L+sj+k];
         if(sum>0) vote[i]++; else vote[j]++;
         p++; }
      vote_max_idx=0; for(i=1;i<nr_class;i++) if(vote[i]>vote[vote_max_idx]) vote_max_idx=i;
      result[d]=(unsigned char)model->label[vote_max_idx]; }
   free(kvalue); free(start); free(vote);
}


/**			PRIVATE FUNTCION
				 * @brief      Computes the metrics for a certain block predictions
				 *
				 * @param      result  Prediction for a block of pixels
         * @param      block_gt Reference data structure containing the labels of the block pixels
         * @param      total Total pixels predicted
				 * @param      correct Number of pixels correctly predicted
				 * @param      length Size of the bloc
				 *
				 * @return     -
				 */
void block_metrics_computation(unsigned char *result, int* block_gt, int *total, int *correct, int length)
{
	bool flag=false;

	for(int i=0;i<length;i++){ //TODO: poner que un determinado porcentaje sea superior a 0
		if(block_gt[i] == 0){
			flag = true;
		}
	}

	if(flag == false){
		for(int i=0;i<length;i++){
			if(block_gt[i] == result[i]){
				(*correct) = (*correct)+1;
			}
			(*total) = (*total)+1;
		}
	}
}


void predict_hsi(command_arguments_struct* command_arguments, struct svm_parameter param, struct svm_model *svm_model, image_struct *image, reference_data_struct *gt_test, char* error, char* message)
{
	int correct = 0;
	int total = 0;
	double e=0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	FILE*output;
	int svm_type=svm_get_svm_type(svm_model);
	unsigned int nr_class=(unsigned int)svm_get_nr_class(svm_model);
	double *prob_estimates=NULL;
	unsigned int j;
	double target_label, predict_label;
	struct svm_node *x;


	output = fopen((char*)get_command_arguments_output_clasftxt(command_arguments),"w");
	if(output == NULL)
	{
		sprintf(error, "can't open output file %s", (char*)get_command_arguments_output_clasftxt(command_arguments));
		print_error((char*)error);
		exit(EXIT_FAILURE);
	}

	//Mapa de clasificacion a generar con la prediccion
	int *classification_map = (int *)malloc(get_reference_data_width(gt_test)*get_reference_data_height(gt_test)*sizeof(int));

	//Vectores de soporte del train
	double *sv=(double *)malloc((svm_model->l)*get_image_bands(image)*sizeof(double));
	for(int k=0;k<(svm_model->l);k++) for(unsigned int j=0;j<get_image_bands(image);j++) sv[k*get_image_bands(image)+j]=svm_model->SV[k][j].value;

	if(param.probability)
	{
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
			printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(svm_model));
		else
		{
			int *labels=(int *) malloc(nr_class*sizeof(int));
			svm_get_labels(svm_model,labels);
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"labels");
			for(j=0;j<nr_class;j++)
				fprintf(output," %d",labels[j]);
			fprintf(output,"\n");
			free(labels);
		}
	}


	if(get_command_arguments_trainpredict_type(command_arguments) == 1){ //prediction by pixel
		x = (struct svm_node *) malloc((get_image_bands(image)+1)*sizeof(struct svm_node));

		for(unsigned int i=0;i<get_reference_data_width(gt_test)*get_reference_data_height(gt_test);i++){

			//Gather array to classify
			for(j=0;j<get_image_bands(image);j++){
				x[j].index = j;
				x[j].value = get_image_data(image)[i*get_image_bands(image)+j];
			}
			target_label = get_reference_data_data(gt_test)[i];

			x[j].index = -1;

			if (param.probability && (svm_type==C_SVC || svm_type==NU_SVC)) //probability prediction type
			{
				predict_label = svm_predict_probability(svm_model,x,prob_estimates);
				fprintf(output,"%g",predict_label);
				for(j=0;j<nr_class;j++)
					fprintf(output," %g",prob_estimates[j]);
				fprintf(output,"\n");
			}
			else //simple prediction type
			{
				//Prediction
				predict_label = svm_predict(svm_model,x); //printf("%f %f\n",target_label, predict_label);
				fprintf(output,"%.17g %.17g\n", target_label, predict_label);
				classification_map[i] = (int)predict_label;
			}
		}
	}else if(get_command_arguments_trainpredict_type(command_arguments) == 2){ //prediction by block
		unsigned char* result;
		double* block_pixels;
		int part_index=0;
		unsigned int p;
		int* block_gt;

		for(p=0;p<(get_reference_data_width(gt_test)*get_reference_data_height(gt_test))/PREDEFINED_BLOCK;p++){

			result = (unsigned char*)malloc(PREDEFINED_BLOCK*sizeof(unsigned char));
			block_pixels = (double*)malloc(PREDEFINED_BLOCK*get_image_bands(image)*sizeof(double));
			block_gt = ( int*)malloc(PREDEFINED_BLOCK*sizeof(int));
			for(unsigned int b1=0;b1<PREDEFINED_BLOCK;b1++){
				for(unsigned int b2=0;b2<get_image_bands(image);b2++){
					block_pixels[ b1*get_image_bands(image)+b2 ] = get_image_data(image)[ part_index ];

					part_index++;
				}
				block_gt[ b1 ] = get_reference_data_data(gt_test)[ p*PREDEFINED_BLOCK+b1 ];
			}
			svm_predict_block(svm_model, sv, block_pixels, result, PREDEFINED_BLOCK, 3, get_image_bands(image));

			block_metrics_computation(result, block_gt, &total, &correct, PREDEFINED_BLOCK);

			for(int b1=0;b1<PREDEFINED_BLOCK;b1++){
				fprintf(output,"%d\n",result[b1]); //TODO: target label
				classification_map[p*PREDEFINED_BLOCK+b1] = result[b1];
			}
			free(result);
			free(block_pixels);
			free(block_gt);
		}
		unsigned int last_block;
		if((last_block = (get_reference_data_width(gt_test)*get_reference_data_height(gt_test)) % PREDEFINED_BLOCK) != 0){
			result = (unsigned char*)malloc(last_block*sizeof(unsigned char));
			block_pixels = (double*)malloc(last_block*get_image_bands(image)*sizeof(double));
			block_gt = ( int*)malloc(PREDEFINED_BLOCK*sizeof( int));

			for(unsigned int b1=0;b1<last_block;b1++){
				for(unsigned int b2=0;b2<get_image_bands(image);b2++){
					block_pixels[ b1*get_image_bands(image)+b2 ] = get_image_data(image)[ part_index ];

					part_index++;
				}
				block_gt[ b1 ] = get_reference_data_data(gt_test)[ p*last_block+b1 ];
			}
			svm_predict_block(svm_model, sv, block_pixels, result, last_block, 3, get_image_bands(image));

			block_metrics_computation(result, block_gt, &total, &correct, last_block);

			for(unsigned int b1=0;b1<last_block;b1++){
				fprintf(output,"%d\n",result[b1]);
				classification_map[p*PREDEFINED_BLOCK+b1] = result[b1];
			}
			free(result);
			free(block_pixels);
			free(block_gt);
		}
	}


	// Classification map saving
	classification_map_ppm((char*)get_command_arguments_output_clasfmap(command_arguments), classification_map, get_reference_data_width(gt_test), get_reference_data_height(gt_test), error, message);

	//Accuracy computation
	if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		printf("Mean squared error = %g (regression)\n",e/total);
		printf("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else{
		// Confusion matrix computation
		confusion_matrix( gt_test, classification_map, NULL );
	}



	if(param.probability)
		free(prob_estimates);

}


void predict_texture(command_arguments_struct* command_arguments, texture_struct* descriptors, struct svm_model *svm_model, segmentation_struct *seg_image, reference_data_struct *gt_test, char* error, char* message)
{
	int j;
	double predict_label;
	int *predict_labels_aux = (int*)malloc(get_descriptors_number_descriptors(descriptors)*sizeof(int));
	struct svm_node *x;

	//Mapa de clasificacion a generar con la prediccion
	int *classification_map = (int *)malloc(get_reference_data_width(gt_test)*get_reference_data_height(gt_test)*sizeof(int));

	//prediction by segment
	x = (struct svm_node *) malloc((get_descriptors_dim_descriptors(descriptors)+1)*sizeof(struct svm_node));

	for(int i=0;i<get_descriptors_number_descriptors(descriptors);i++){

		//Gather array to classify
		for(j=0;j<get_descriptors_dim_descriptors(descriptors);j++){
			x[j].index = j;
			x[j].value = get_descriptors_data(descriptors)[i*get_descriptors_dim_descriptors(descriptors)+j];
		}
		//target_label = get_descriptors_labels(descriptors)[i];

		x[j].index = -1;

		//Prediction
		predict_label = svm_predict(svm_model,x); //printf("%f %f\n",target_label, predict_label);

		predict_labels_aux[i] = (int)predict_label;
	}

	//Classification map: labels per segment
	set_labels_per_segment(seg_image, classification_map, predict_labels_aux, get_descriptors_number_descriptors(descriptors));

	// Classification map saving
	//classification_map_ppm((char*)get_command_arguments_output_clasfmap(command_arguments), classification_map, get_reference_data_width(gt_test), get_reference_data_height(gt_test), error, message);

	// Prediction textfile saving
	//prediction_textfile(classification_map, gt_test, (char*)get_command_arguments_output_clasftxt(command_arguments), error);

	// Confusion matrix computation
	confusion_matrix( gt_test, classification_map, seg_image);
}
