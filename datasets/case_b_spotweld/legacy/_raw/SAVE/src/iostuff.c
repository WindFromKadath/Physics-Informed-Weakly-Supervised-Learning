#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "iostuff.h"
#include "setup.h"
#include <math.h>
#include <R.h>
#include <R_ext/Print.h>

void dprintmat(double **a, int n, int m)
/* 
  prints the nxm matrix a to the screen 
*/
{ 
 int i, j;

 Rprintf("\n");
 for(i=0;i<n;i++) {
   for(j=0;j<m;j++) {
     Rprintf("%f ",a[i][j]);
   }
   Rprintf("\n");
 }
 Rprintf("End of the matrix\n");
}

int dreadmat(char file[], double **a, int n, int m) 
/*
  Reads the file "file" and puts it into the nxm matrix a
*/
{
 FILE *fp;
 int res;

 //Rprintf("Trying to read the file:  %s\n",file);
 if ((fp = fopen(file,"r"))==NULL){
   Rprintf("cannot open file \n");
   return(1);
 }

 if (feof(fp)) Rprintf("eof");
   for(int i=0;i<n;i++){
     for(int j=0;j<m;j++) {
       res = fscanf(fp,"%lf",&a[i][j]);
     }
   }
 fclose(fp);
 //Rprintf("Read of <<%s>> complete\n",file);
 return(0);
}

int dwritemat(char *file,double **a, int n, int m) 
/*
  Writes the nxm matrix a to "file" 
*/
{
 FILE *fp;
 /* Rprintf("Trying to write the file:  %s\n",file); */
 if ((fp = fopen(file,"w"))==NULL){
   Rprintf("cannot open %s \n",file);
   return(1);
 }

 for(int i=0;i<n;i++){
   for(int j=0;j<m;j++) {
     fprintf(fp,"%g ",a[i][j]);
   }
   fprintf(fp,"\n");
 }
 fclose(fp);
 //Rprintf("<<%s>> written\n",file);
 return (0);
}

int iappendmat(char *file, int **a, int n, int m) 
/*
  Writes the nxm matrix a to "file" 
*/
{
 FILE *fp;
 /* Rprintf("Trying to write the file:  %s\n",file); */
 if ((fp = fopen(file,"a"))==NULL){
   Rprintf("cannot open %s \n",file);
   return(1);
 }

 for(int i=0;i<n;i++){
   for(int j=0;j<m;j++) {
     fprintf(fp,"%d ",a[i][j]);
   }
   fprintf(fp,"\n");
 }
 fclose(fp);
 //Rprintf("<<%s>> appended\n",file);
 return (0);
}

int appendvec(char file[], float *a, int n) 
/*
  Writes the vector a[0..n-1] to "file" 
*/
{
 FILE *fp;
 /* Rprintf("Trying to write the file:  %s\n",file); */
 if ((fp = fopen(file,"a"))==NULL){
   Rprintf("cannot open %s \n",file);
   return(1);
 }

 for(int i=0;i<n;i++){
     fprintf(fp,"%g ",a[i]);
 }
 fprintf(fp,"\n");
 fclose(fp);
 //Rprintf("<<%s>> appended\n",file);
 return(0);
}
 
int iappendvec(char *file, int *a, int n) 
/*
  Writes the vector a[0..n-1] to "file" 
*/
{
 FILE *fp;
 //Rprintf("Trying to write the file:  %s\n",file);
 if ((fp = fopen(file,"a"))==NULL){
   Rprintf("cannot open %s \n",file);
   return(1);
 }

 for(int i=0;i<n;i++){
     fprintf(fp,"%d ",a[i]);
     if(((i+1)%10) == 0) fprintf(fp,"\n");
 }
 fprintf(fp,"\n");
 fclose(fp);
 //Rprintf("<<%s>> appended\n",file);
 return (0);
}
 
int dappendvec(char *file, double *a, int n) 
/*
  Writes the vector a[0..n-1] to "file" 
*/
{
 FILE *fp;
 //Rprintf("Trying to write the file:  %s\n",file);
 if ((fp = fopen(file,"a"))==NULL){
   Rprintf("cannot open %s \n",file);
   return(1);
 }

 for(int i=0;i<n;i++){
     fprintf(fp,"%g ",a[i]);
 }
 fprintf(fp,"\n");
 fclose(fp);
 //Rprintf("<<%s>> appended\n",file);
 return (0);
}

int dappendvecnotverb(char *file, double *a, int n) 
/***************************************************************************
  Writes the vector a[0..n-1] to "file" 
  a    *float
  n    int  # of elements
*************************************************************************/
{
  //Rprintf("About to write in %s \n",file);
  FILE *fp;
 //Rprintf("Trying to write the file:  %s\n",file);
 if ((fp = fopen(file,"a"))==NULL){
   Rprintf("cannot open %s \n",file);
   return(1);
 }

 for(int i=0;i<n;i++){
     fprintf(fp,"%lf ",a[i]);
 }
 fprintf(fp,"\n");
 fclose(fp);
 //Rprintf("<<%s>> appended\n",file);
 return (0);
}
 
int appendmat(char *file, float **a, int n, int m) 
/*
  Writes the matrix a[0..n-1][0..m-1] to "file" 
*/
{
 FILE *fp;
 //Rprintf("Trying to write the file:  %s\n",file);
 if ((fp = fopen(file,"a"))==NULL){
   Rprintf("cannot open %s \n",file);
   return(1);
 }

 for(int i=0;i<n;i++){
   for(int j=0;j<m;j++) {
     fprintf(fp,"%g ",a[i][j]);
   }
   fprintf(fp,"\n");
 }
 fclose(fp);
 //Rprintf("<<%s>> appended\n",file);
 return (0);
}
 
int dappendmat(char *file, double **a, int n, int m) 
/*
  Writes the matrix a[0..n-1][0..m-1] to "file" 
*/
{
 FILE *fp;
 //Rprintf("Trying to write the file:  %s\n",file);
 if ((fp = fopen(file,"a"))==NULL){
   Rprintf("cannot open %s \n",file);
   return(1);
 }

 for(int i=0;i<n;i++){
   for(int j=0;j<m;j++) {
     fprintf(fp,"%g ",a[i][j]);
   }
   fprintf(fp,"\n");
 }
 fclose(fp);
 //Rprintf("<<%s>> appended\n",file);
 return (0);
}
 
void iprintmat(int **a, int n, int m)
/*
  prints the matrix a to the screen 
*/
{ 
 Rprintf("\n");
 for(int i=0;i<n;i++) {
   for(int j=0;j<m;j++) {
     Rprintf("%d ",a[i][j]);
   }
   Rprintf("\n");
 }
}
 

int ireadmat(char file[], int **a, int n, int m) 
/*
  Reads the file "file" and puts it into the matrix a[1..n][1..m]
*/
{
 FILE *fp;
 int res;
 //Rprintf("Trying to read the file:  %s\n",file);
 if ((fp = fopen(file,"r"))==NULL){
   Rprintf("cannot open file \n");
   return(1);
 }

 if (feof(fp)) Rprintf("eof");
   for(int i=0;i<n;i++){
     for(int j=0;j<m;j++) {
       res = fscanf(fp,"%d",&a[i][j]);
       /* Rprintf("%f",a[i][j]); */
     }
   }
 fclose(fp);
 //Rprintf("Read of <<%s>> complete\n",file);
 return(0);
}

int iwritemat(char *file, int **a, int n, int m) 
/*
  Writes the matrix a[1..n][1..m] to "file" 
*/
{
 FILE *fp;
 //Rprintf("Trying to write the file:  %s\n",file);
 if ((fp = fopen(file,"w"))==NULL){
   Rprintf("cannot open %s \n",file);
   return(1);
 }

 for(int i=0;i<n;i++){
   for(int j=0;j<m;j++) {
     fprintf(fp,"%d ",a[i][j]);
   }
   fprintf(fp,"\n");
 }
 fclose(fp);
 //Rprintf("<<%s>> written\n",file);
 return(0);	
}

void printmat(float **a, int n, int m)
/*
  prints the matrix a to the screen 
*/
{ 

 Rprintf("\n");
 for(int i=0;i<n;i++) {
   for(int j=0;j<m;j++) {
     Rprintf("%f ",a[i][j]);
   }
   Rprintf("\n");
 }
}
 

int readmat(char file[], float **a, int n, int m) 
/*
  Reads the file "file" and puts it into the matrix a[1..n][1..m]
*/
{
 FILE *fp;
 int res;
 //Rprintf("Trying to read the file:  %s\n",file);
 if ((fp = fopen(file,"r"))==NULL){
   Rprintf("cannot open file \n");
   return (1);
 }

 if (feof(fp)) Rprintf("eof");
   for(int i=0;i<n;i++){
     for(int j=0;j<m;j++) {
       res = fscanf(fp,"%f",&a[i][j]);
       /* Rprintf("%f",a[i][j]); */
     }
   }
 fclose(fp);
 //Rprintf("Read of <<%s>> complete\n",file);
 return (0);
}

void uliprintmat(unsigned long int **a, int n, int m)
/*
  prints the matrix a to the screen 
*/
{ 

 Rprintf("\n");
 for(int i=0;i<n;i++) {
   for(int j=0;j<m;j++) {
     Rprintf("%u ",a[i][j]);
   }
   Rprintf("\n");
 }
}

int uliwritemat(char *file, unsigned long int **a, int n, int m) 
/*
  Writes the matrix a[1..n][1..m] to "file" 
*/
{
 FILE *fp;
 //Rprintf("Trying to write the file:  %s\n",file);
 if ((fp = fopen(file,"w"))==NULL){
   Rprintf("cannot open %s \n",file);
   return (1);
 }

 for(int i=0;i<n;i++){
   for(int j=0;j<m;j++) {
     fprintf(fp,"%lu ",a[i][j]);
   }
   fprintf(fp,"\n");
 }
 fclose(fp);
 //Rprintf("<<%s>> written\n",file);
 return (0);
}

int writemat(char *file, float **a, int n, int m) 
/*
  Writes the matrix a[1..n][1..m] to "file" 
*/
{
 FILE *fp;
 //Rprintf("Trying to write the file:  %s\n",file);
 if ((fp = fopen(file,"w"))==NULL){
   Rprintf("cannot open %s \n",file);
   return(1);
 }

 for(int i=0;i<n;i++){
   for(int j=0;j<m;j++) {
     fprintf(fp,"%f ",a[i][j]);
   }
   fprintf(fp,"\n");
 }
 fclose(fp);
 //Rprintf("<<%s>> written\n",file);
 return (0);
}

void dprintvec(double *v, int n)
/*
  prints the vector a to the screen 
  v  *double  
  n  int  # of elements
*/
{ 

 for(int i=0;i<n;i++) {
   Rprintf("%g ",v[i]);
 }
 Rprintf("\n");
}
 
int dreadvec(char file[], double *v, int n) 
/*
  Reads the file "file" and puts it into the vector v[0..n-1]

*/
{
 FILE *fp;
 int res;
 //Rprintf("Trying to read the file:  %s\n",file);
 if ((fp = fopen(file,"r"))==NULL){
   Rprintf("cannot open file \n");
   return(1);
 }

 if (feof(fp)) Rprintf("eof");
   for(int i=0;i<n;i++){
       res = fscanf(fp,"%lf",&v[i]);
        //Rprintf("%lf ",v[i]);
   }
 fclose(fp);
 //Rprintf("Read of <<%s>> complete\n",file);
 return (0);
}
 
void iprintvec(int *v, int n)
/*
  prints the vector a to the screen 
*/
{ 

 Rprintf("\n");
 for(int i=0;i<n;i++) {
     Rprintf("%d ",v[i]);
 }
 Rprintf("\n");
}

int ireadvec(char file[], int *v, int n) 
/*
  Reads the file "file" and puts it into the vector v[1..n]
*/
{
 FILE *fp;
 int res;
 //Rprintf("Trying to read the file:  %s\n",file);
 if ((fp = fopen(file,"r"))==NULL){
   Rprintf("cannot open file \n");
   return(1);
 }

 if (feof(fp)) Rprintf("eof");
   for(int i=0;i<n;i++){
       res = fscanf(fp,"%d",&v[i]);
       /* Rprintf("%f",v[i]); */
   }
 fclose(fp);
 // Rprintf("Read of <<%s>> complete\n",file);
 return (0);
}

int iwritevec(char file[], int *v, int n) 
/*
  Writes the vector v[1..n] to the file "file" 
*/
{
 FILE *fp;
 //Rprintf("Trying to write file:  %s\n",file);
 if ((fp = fopen(file,"w"))==NULL){
   Rprintf("cannot open %s \n",file);
   return (1);
 }

 for(int i=0;i<n;i++){
     fprintf(fp,"%d\n",v[i]);
     /* Rprintf("%d",v[i]); */
 }
 fclose(fp);
 //Rprintf("<<%s>> written\n",file);
 return (0);
}

void printvec(float *v, int n)
/*
  prints the vector a to the screen 
*/
{ 

 for(int i=0;i<n;i++) {
     Rprintf("%f ",v[i]);
 }
 Rprintf("\n");
}

int readvec(char file[], float *v, int n) 
/*
  Reads the file "file" and puts it into the vector v[0..n-1]
*/
{
 FILE *fp;
 int res;
 //Rprintf("Trying to read the file:  %s\n",file);
 if ((fp = fopen(file,"r"))==NULL){
   Rprintf("cannot open file \n");
   return (1);
 }

 if (feof(fp)) Rprintf("eof");
 for(int i=0;i<n;i++){
     res = fscanf(fp,"%f",&v[i]);
     /* Rprintf("%f",v[i]); */
 }
 fclose(fp);
 //Rprintf("Read of <<%s>> complete\n",file);
 return (0);
}

int uliwritevec(char file[], unsigned long int *v,int n) 
/*
  Writes the vector v[1..n] to the file "file" 
*/
{
 FILE *fp;
 //Rprintf("Trying to write file:  %s\n",file);
 if ((fp = fopen(file,"w"))==NULL){
   Rprintf("cannot open %s \n",file);
   return (1);
 }

 for(int i=0;i<n;i++){
     fprintf(fp,"%lu\n",v[i]);
     /* Rprintf("%u",v[i]); */
 }
 fclose(fp);
 //Rprintf("<<%s>> written\n",file);
 return (0);
}
 
int uliappendvec(char file[], unsigned long int *v, int n) 
/*
  Writes the vector v[1..n] to the file "file" 
*/
{
 FILE *fp;
 //Rprintf("Trying to write file:  %s\n",file);
 if ((fp = fopen(file,"a"))==NULL){
   Rprintf("cannot open %s \n",file);
   return (1);
 }

 for(int i=0;i<n;i++){
     fprintf(fp,"%lu ",v[i]);
     if(((i+1)%10) == 0) fprintf(fp,"\n");
     /* Rprintf("%u",v[i]); */
 }
 fclose(fp);
 //Rprintf("<<%s>> written\n",file);
 return (0);
}

int writevec(char file[], float *v, int n) 
/*
  Writes the vector v[1..n] to the file "file" 
*/
{
 FILE *fp;
 //Rprintf("Trying to write file:  %s\n",file);
 if ((fp = fopen(file,"w"))==NULL){
   Rprintf("cannot open %s \n",file);
   return (1);
 }

 for(int i=0;i<n;i++){
     fprintf(fp,"%f\n",v[i]);
     /* Rprintf("%f",v[i]); */
 }
 fclose(fp);
 //Rprintf("<<%s>> written\n",file);
 return (0);
}

int dwritevec(char file[], double *v, int n) 
/*
  Writes the vector v[1..n] to the file "file" 
*/
{
 FILE *fp;
 //Rprintf("Trying to write file:  %s\n",file);
 if ((fp = fopen(file,"w"))==NULL){
   Rprintf("cannot open %s \n",file);
   return (1);
 }

 for(int i=0;i<n;i++){
     fprintf(fp,"%f\n",v[i]);
     /* Rprintf("%f",v[i]); */
 }
 fclose(fp);
 //Rprintf("<<%s>> written\n",file);
 return (0);
}

void iprinta3(int ***a, int n1, int n2, int n3)
/*
  prints a to the screen 
*/
{ 

 for(int i1=0;i1<n1;i1++){
 Rprintf("\n");
   for(int i2=0;i2<n2;i2++){
     for(int i3=0;i3<n3;i3++){
       Rprintf("%d ",a[i1][i2][i3]);
     }
     Rprintf("\n");
   }
 }
}

void printa3(float ***a, int n1, int n2, int n3)
/*
  prints a to the screen 
*/
{ 

 for(int i1=0;i1<n1;i1++){
 Rprintf("\n");
   for(int i2=0;i2<n2;i2++){
     for(int i3=0;i3<n3;i3++){
       Rprintf("%f ",a[i1][i2][i3]);
     }
     Rprintf("\n");
   }
 }
}

int ireada3(char file[], int ***a, int n1, int n2, int n3) 
/*
  Reads the file "file" and puts it into the 
       array a[0..n1-1][0..n2-1][0..n3-1]
*/
{
 FILE *fp;
 int res;
 //Rprintf("Trying to read the file:  %s\n",file);
 if ((fp = fopen(file,"r"))==NULL){
   Rprintf("cannot open file \n");
   return (1);
 }

 if (feof(fp)) Rprintf("eof");
   for(int i1=0;i1<n1;i1++){
     for(int i2=0;i2<n2;i2++){
       for(int i3=0;i3<n3;i3++){
         res = fscanf(fp,"%d",&a[i1][i2][i3]);
         /* Rprintf("%d ",a[i1][i2][i3]); */ 
       }
     }
   }
 fclose(fp);
 //Rprintf("Read of <<%s>> complete\n",file);
 return (0);
}

int reada3(char file[], float ***a, int n1, int n2, int n3) 
/*
  Reads the file "file" and puts it into the 
       array a[0..n1-1][0..n2-1][0..n3-1]
*/
{
 FILE *fp;
 int res;
 //Rprintf("Trying to read the file:  %s\n",file);
 if ((fp = fopen(file,"r"))==NULL){
   Rprintf("cannot open file \n");
   return(1);
 }

 if (feof(fp)) Rprintf("eof");
   for(int i1=0;i1<n1;i1++){
     for(int i2=0;i2<n2;i2++){
       for(int i3=0;i3<n3;i3++){
         res = fscanf(fp,"%f",&a[i1][i2][i3]);
         /* Rprintf("%f ",a[i1][i2][i3]); */ 
       }
     }
   }
 fclose(fp);
 //Rprintf("Read of <<%s>> complete\n",file);
 return (0);
}

int writea3(char file[], float ***a, int n1, int n2, int n3)
/*
  prints a to file, 20 cols at a time
*/
{
 FILE *fp;
 //Rprintf("Trying to write file:  %s\n",file);
 if ((fp = fopen(file,"w"))==NULL){
   Rprintf("cannot open %s \n",file);
   return(1);
 }

 int count = 0;

 for(int i1=0;i1<n1;i1++){
   for(int i2=0;i2<n2;i2++){
     for(int i3=0;i3<n3;i3++){
       fprintf(fp,"%g ",a[i1][i2][i3]);
       count ++;
       if(count % 20 == 0) fprintf(fp,"\n");
     }
   }
 }
 fclose(fp);
 //Rprintf("<<%s>> written\n",file);
 return (0);
}

int dwritea3(char file[], double ***a, int n1, int n2, int n3)
/*
  prints a to file, 20 cols at a time
*/
{
 FILE *fp;
 //Rprintf("Trying to write file:  %s\n",file);
 if ((fp = fopen(file,"w"))==NULL){
   Rprintf("cannot open %s \n",file);
   return (1);
 }

 int count = 0;

 for(int i1=0;i1<n1;i1++){
   for(int i2=0;i2<n2;i2++){
     for(int i3=0;i3<n3;i3++){
       fprintf(fp,"%g ",a[i1][i2][i3]);
       count ++;
       if(count % 20 == 0) fprintf(fp,"\n");
     }
   }
 }
 fclose(fp);
 //Rprintf("<<%s>> written\n",file);
 return (0);
}

int appenda3(char file[], float ***a, int n1, int n2, int n3)
/*
  appends a to file, 20 cols at a time
*/
{
 FILE *fp;
 //Rprintf("Trying to write file:  %s\n",file);
 if ((fp = fopen(file,"a"))==NULL){
   REprintf("cannot open %s \n",file);
   error(".\n");
 }

 int count = 0;

 for(int i1=0;i1<n1;i1++){
   for(int i2=0;i2<n2;i2++){
     for(int i3=0;i3<n3;i3++){
       fprintf(fp,"%g ",a[i1][i2][i3]);
       count ++;
       if(count % 20 == 0) fprintf(fp,"\n");
     }
   }
 }
 fclose(fp);
 //Rprintf("<<%s>> appended\n",file);
 return (0);
}

double * get_datacode (char home[], int nm){
	char strtmp[100];
	char datacode[100] = "model_data.dat"; /* file where the code data are */
	//Rprintf("datacode is: %s\n",datacode);
	strcpy(strtmp,home);
	//Rprintf("strtmp is: %s\n",strtmp);
	strcat(strtmp,datacode);
	//Rprintf("strtmp is: %s\n",strtmp);
	strcpy(datacode,strtmp);
	/* file where the code data are */
	
	
	double * res;
	res = dvector(0,nm);
	int auxError = 0;
	auxError = dreadvec(datacode,res,nm); /* read model data */
	//if (screen !=0) Rprintf("yM[0]%lf\n",yM[0]);
	if (auxError!=0) Rprintf("Error while getting access to the file where the code data are =%s\n",datacode);
	return res;
}

int * get_inputsrep (char home[], int nf){
	char strtmp[100];
	char inputsrep[100] = "field_nreps.dat"; /* file containing the number of replicates at each
											  unique design point */
	strcpy(strtmp,home);
	strcat(strtmp,inputsrep);
	strcpy(inputsrep,strtmp);
	
	int * res;
	res = ivector(0,nf);
	int auxError = 0;
	auxError = ireadvec(inputsrep,res,nf); /* read vector of number of replicates */
	if (auxError!=0) Rprintf("Error while getting access to the file where the number of replicates =%s\n",inputsrep);
	return res;
}

double * get_mlethetaF (char home[], int pcont){
	char strtmp[100];
	char fmle[100] = "thetaF_mle.dat"; /* file containing the GASP mle of thetaF*/
	strcpy(strtmp,home);
	strcat(strtmp,fmle);
	strcpy(fmle,strtmp);
	
	double * res;
	res = dvector(0,2*(pcont)+2);
	int auxError = 0;
	auxError = dreadvec(fmle,res,2*(pcont)+2);
	if (auxError!=0) Rprintf("Error while getting access to the file where the GASP mle of thetaF =%s\n",fmle);
	return res;
}

double * get_mlethetaM (char home[], int p){
	char strtmp[100];
	char fileM[100] = "thetaM_mle.dat"; /* file containing the stage I GASP mle of thetaM */
	strcpy(strtmp,home);
	strcat(strtmp,fileM);
	strcpy(fileM,strtmp);
	
	double * res;
	res = dvector(0,2*p+1);
	int auxError = 0;
	auxError = dreadvec(fileM,res,2*p+1);
	if (auxError!=0) Rprintf("Error while getting access to the file where the GASP mle of thetaM =%s\n",fileM);
	return res;
}

double * get_mlethetaL (char home[], int q){
	char strtmp[100];
	char fileL[100] = "thetaL_mle.dat"; /* file contaning the stage I GASP mle of thetaL */
	strcpy(strtmp,home);
	strcat(strtmp,fileL);
	strcpy(fileL,strtmp);
	
	double * res;
	res = dvector(0,q);
	int auxError = 0;
	auxError = dreadvec(fileL,res,q);
	if (auxError!=0) Rprintf("Error while getting access to the file where the GASP mle of thetaL =%s\n",fileL);
	return res;
}

double * get_datafield (char home[], int *nrep, int nf){
	char strtmp[100];
	char datafield[100] = "field_data.dat"; /* file where the field data are */
	strcpy(strtmp,home);
	strcat(strtmp,datafield);
	strcpy(datafield,strtmp);
	
	int nftot = 0;
	/* total number of field observations (this number includes 
	 the replicates)
	 compute total number of field observations */
	for(int i=0;i<nf;i++){
		nftot = nftot+nrep[i];
	}
	
	double * auxres;
	auxres = dvector(0,nftot);
	int auxError = 0;
	auxError = dreadvec(datafield,auxres,nftot); /* read field data */
	if (auxError!=0) Rprintf("Error while getting access to the file where the field data are =%s\n",datafield);
	double * res;
	res = dvector(0,nf);
	int ntmp=0;
	for(int i=0;i<nf;i++){
		res[i]=0.0;
		for(int j=0;j<nrep[i];j++){
			res[i]=res[i]+auxres[ntmp];
			ntmp=ntmp+1;
		}
		res[i]=res[i]/(double)nrep[i];
	}
	free_dvector(auxres,0,nftot);
	
	return res;
}

double * get_datafield_s2F (char home[], int *nrep, int nf, double *s2F){
	// same as get_datafield but it returns also the data -- sum of squared deviations 
	//from individual means */

	char strtmp[100];
	char datafield[100] = "field_data.dat"; /* file where the field data are */
	strcpy(strtmp,home);
	strcat(strtmp,datafield);
	strcpy(datafield,strtmp);
	
	int nftot = 0;
	/* total number of field observations (this number includes 
	 the replicates)
	 compute total number of field observations */
	for(int i=0;i<nf;i++){
		nftot = nftot+nrep[i];
	}
	
	double * auxres;
	auxres = dvector(0,nftot);
	int auxError = 0;
	auxError = dreadvec(datafield,auxres,nftot); /* read field data */
	if (auxError!=0) Rprintf("Error while getting access to the file where the field data are =%s\n",datafield);
	double * res;
	res = dvector(0,nf);
	int ntmp=0;
	*s2F = 0.0;
	for(int i=0;i<nf;i++){
		res[i]=0.0;
		for(int j=0;j<nrep[i];j++){
			res[i]=res[i]+auxres[ntmp];
			*s2F=*s2F+pow(auxres[ntmp],2.);
			ntmp=ntmp+1;
		}
		res[i]=res[i]/(double)nrep[i];
		*s2F=*s2F-(double)nrep[i]*pow(res[i],2.);
	}
	free_dvector(auxres,0,nftot);
	
	return res;
}

double ** get_inputscode (char home[], int nm, int p){
	char strtmp[100];
	char inputscode[100] = "model_inputs.dat"; /* file where the code data inputs are */
	strcpy(strtmp,home);
	strcat(strtmp,inputscode);
	strcpy(inputscode,strtmp);
	
	double ** auxres;
	auxres = darray2(nm,p);
	int auxError = 0;
	auxError = dreadmat(inputscode,auxres,nm,p); /* read model design set */
	if (auxError!=0) Rprintf("Error while getting access to the file where the model design set is =%s\n",inputscode);
	double **res;
	res = darray2(p,nm);
	for(int i=0;i<p;i++){
		for(int j=0;j<nm;j++){
			res[i][j]=auxres[j][i];
		}
	}
	free_darray2(auxres,nm,p);
	//Rprintf("code data inputs read and loaded from %s\n",inputscode);
	return res;
}

double ** get_inputsfield (char home[], int nf, int pcont){
	char strtmp[100];
	char inputsfield[100] = "field_unique.dat"; /* file where the field data inputs are */
	strcpy(strtmp,home);
	strcat(strtmp,inputsfield);
	strcpy(inputsfield,strtmp);

	double ** auxres;
	auxres = darray2(nf,pcont);
	int auxError = 0;
	auxError = dreadmat(inputsfield,auxres,nf,pcont); /* read field design set */
	if (auxError!=0) Rprintf("Error while getting access to the file where the field design set is =%s\n",inputsfield);
	double ** res;
	res = darray2(pcont,nf);
	for(int i=0;i<pcont;i++){
		for(int j=0;j<nf;j++){
			res[i][j]=auxres[j][i];
		}
	}
	free_darray2(auxres,nf,pcont);
	
	return res;
}

double ** get_designMatrix (char home[], int nf, int nm, int q){
	char strtmp[100];
	char designM[100] = "mcmc.field.design.M.matrix.dat";
	strcpy(strtmp,home);
	strcat(strtmp,designM);
	strcpy(designM,strtmp);
	
	double ** auxresM;
	auxresM = darray2(nm,q);
	int auxError = 0;
	auxError = dreadmat(designM,auxresM,nm,q);  /* read design matrix for code */
	if (auxError!=0) Rprintf("Error while getting access to the file where the design matrixM is =%s\n",designM);

	char designF[100] = "mcmc.field.design.F.matrix.dat";
	strcpy(strtmp,home);
	strcat(strtmp,designF);
	strcpy(designF,strtmp);

	double ** auxresF;
	auxresF = darray2(nf,q);
	auxError = dreadmat(designF,auxresF,nf,q);  /* read design matrix for code */
	if (auxError!=0) Rprintf("Error while getting access to the file where the design matrixF is =%s\n",designF);
	
	int n, i, j;
	n = nf+nm;
	double **res;
	res = darray2(n,q);
	for(j=0; j<q; j++){
		for(i=0; i<nm; i++){
			res[i][j]=auxresM[i][j];
		}
		for(i=0; i<nf; i++){
			res[i+nm][j]=auxresF[i][j];
		}
	}
	
	free_darray2(auxresM,nm,q);	
	free_darray2(auxresF,nf,q);
	
	return res;
}

double ** get_ubounds (char home[], int pstar){
	char strtmp[100];
	/* read the matrix with bounds on u */
	/* and the matrix specifying the priors and the bounds */
	char fboundsU[100] = "bounds.dat";
	
	double ** res;
	if (pstar != 0){
		strcpy(strtmp,home);
		strcat(strtmp,fboundsU);
		strcpy(fboundsU,strtmp);

		/*	*ubounds = (double **) malloc((unsigned)pstar * sizeof(double *));
		(*ubounds)[0] = (double *) malloc((unsigned)(pstar*5) * sizeof(double));
		for(i=1;i<pstar;i++) (*ubounds)[i] = (*ubounds)[i-1] + 5;
		 */
		res = darray2(pstar,5);
		int auxError = 0;
		auxError = dreadmat(fboundsU,res,pstar,5);
		if (auxError!=0) Rprintf("Error while getting access to the file where the ubounds are =%s\n",fboundsU);
	} else Rprintf ("Warning!!: trying to read priors for u when there are not calibration parameters\n");
	return res;
}
