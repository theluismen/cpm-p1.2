#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <strings.h>
#include <assert.h>
#include <omp.h>

#define N 8000L
#define ND N*N/100

typedef struct {
    int i,j,v;
} tmd;

int A[N][N],B[N][N],C[N][N],C1[N][N],C2[N][N];
int jBD[N+1];
int iAD[N+1]; // NUEVO: Índice para las filas de AD
tmd AD[ND],BD[ND],CD[N*N];

long long Suma;

int cmp_fil(const void *pa, const void *pb) {
    tmd * a = (tmd*)pa;
    tmd * b = (tmd*)pb;
    if (a->i > b->i) return(1);
    else if (a->i < b->i) return (-1);
    else return (a->j - b->j);
}

int cmp_col(const void *pa, const void *pb) {
    tmd * a = (tmd*)pa;
    tmd * b = (tmd*)pb;
    if (a->j > b->j) return(1);
    else if (a->j < b->j) return (-1);
    else return (a->i - b->i);
}

int main()
{
    int i,j,k,neleC;
    
    bzero(C,sizeof(int)*(N*N));
    bzero(C1,sizeof(int)*(N*N));
    bzero(C2,sizeof(int)*(N*N));
     
    // Inicialización rand()
    for(k=0;k<ND;k++) {
        AD[k].i=rand()%(N-1);
        AD[k].j=rand()%(N-1);
        AD[k].v=rand()%100+1;
        while (A[AD[k].i][AD[k].j]) {
            if(AD[k].i < AD[k].j) AD[k].i = (AD[k].i + 1)%N;
            else AD[k].j = (AD[k].j + 1)%N;
        }
        A[AD[k].i][AD[k].j] = AD[k].v;
    }
    qsort(AD,ND,sizeof(tmd),cmp_fil);

    for(k=0;k<ND;k++) {
        BD[k].i=rand()%(N-1);
        BD[k].j=rand()%(N-1);
        BD[k].v=rand()%100+1;
        while (B[BD[k].i][BD[k].j]) {
            if(BD[k].i < BD[k].j) BD[k].i = (BD[k].i + 1)%N;
            else BD[k].j = (BD[k].j + 1)%N;
        }
        B[BD[k].i][BD[k].j] = BD[k].v;
    }
    qsort(BD,ND,sizeof(tmd),cmp_col);
    
    // Calcul dels index de les columnes de B
    k=0;
    for (j=0; j<N+1; j++) {
      while (k < ND && j>BD[k].j) k++;
      jBD[j] = k;
    }

    // EL TRUCO MAGICO: Calcul dels index de les files de A
    k=0;
    for (i=0; i<N+1; i++) {
        while (k < ND && i>AD[k].i) k++;
        iAD[i] = k;
    }

    neleC = 0;

    // ================================================================
    // ZONA PARALELA ALGORÍTMICAMENTE OPTIMIZADA
    // ================================================================
    #pragma omp parallel
    {
        int VBcol_priv[N];
        tmd col_buffer[N];
        for (int j=0; j<N; j++) VBcol_priv[j] = 0;

        // ----------------------------------------------------------------
        // 1. Matriu dispersa per matriu (Loop Inversion + Accesos Lineales)
        // ----------------------------------------------------------------
        #pragma omp for schedule(static)
        for (int rC = 0; rC < N; rC++) {
            // Recorremos SOLO los elementos no cero de esta fila
            for (int k = iAD[rC]; k < iAD[rC+1]; k++) {
                int rB = AD[k].j;
                int val = AD[k].v;
                
                // Usamos punteros para ayudar al compilador a volar (Stride-1)
                int *pC1 = C1[rC];
                int *pB = B[rB];
                for (int c = 0; c < N; c++) {
                    pC1[c] += val * pB[c];
                }
            }
        }

        // ----------------------------------------------------------------
        // 2 y 3. FUSIÓN TOTAL (Evitamos escribir en memoria inútilmente)
        // ----------------------------------------------------------------
        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            // Expandir Columna i de B
            for (int k = jBD[i]; k < jBD[i+1]; k++) {
                VBcol_priv[BD[k].i] = BD[k].v;
            }
            
            int col_count = 0;
            // Para cada fila de la matriz resultante (C)
            for (int rC = 0; rC < N; rC++) {
                int sum = 0;
                
                // Producto punto exacto (solo elementos no cero)
                for (int k = iAD[rC]; k < iAD[rC+1]; k++) {
                    sum += AD[k].v * VBcol_priv[AD[k].j];
                }
                
                // Si hay resultado, escribimos directo a Densa y Dispersa
                if (sum != 0) {
                    C2[rC][i] = sum;
                    
                    col_buffer[col_count].i = rC;
                    col_buffer[col_count].j = i;
                    col_buffer[col_count].v = sum;
                    col_count++;
                }
            }
            
            // Volcado ultra rápido
            if (col_count > 0) {
                int start_idx;
                #pragma omp atomic capture
                {
                    start_idx = neleC;
                    neleC += col_count;
                }
                for (int c = 0; c < col_count; c++) {
                    CD[start_idx + c] = col_buffer[c];
                }
            }
            
            // Fast clear de B
            for (int k = jBD[i]; k < jBD[i+1]; k++) {
                VBcol_priv[BD[k].i] = 0;
            }
        }

        // ----------------------------------------------------------------
        // Comprobación
        // ----------------------------------------------------------------
        #pragma omp for schedule(static)
        for (int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                if (C2[i][j] != C1[i][j])
                    printf("Diferencies C1 i C2 pos %d,%d: %d != %d\n",i,j,C1[i][j],C2[i][j]);
            }
        }
    } 

    Suma = 0;
    #pragma omp parallel for reduction(+:Suma) schedule(static)
    for(k=0; k<neleC; k++) {
        Suma += CD[k].v;
        if (CD[k].v != C1[CD[k].i][CD[k].j])
            printf("Diferencies C1 i CD a i:%d,j:%d,v%d, k:%d, vd:%d\n",CD[k].i,CD[k].j,C1[CD[k].i][CD[k].j],k,CD[k].v);
    }
     
    printf ("\nNumero elements de la matriu dispersa C %d\n",neleC);   
    printf("Suma dels elements de C %lld \n",Suma);
    exit(0);
}