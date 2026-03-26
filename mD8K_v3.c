#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <strings.h>
#include <assert.h>
#include <omp.h>

#define N 8000L
#define ND N*N/100

typedef struct {
    int i, j, v;
} tmd;

int A[N][N], B[N][N], C[N][N], C1[N][N], C2[N][N];
int jBD[N + 1];
tmd AD[ND], BD[ND], CD[N * N];

long long Suma;

int cmp_fil ( const void *pa, const void *pb )
{
    tmd *a = (tmd *) pa;
    tmd *b = (tmd *) pb;

    if ( a->i > b->i )
        return 1;
    else if ( a->i < b->i )
        return -1;
    else
        return a->j - b->j;
}

int cmp_col ( const void *pa, const void *pb )
{
    tmd *a = (tmd *)pa;
    tmd *b = (tmd *)pb;

    if ( a->j > b->j )
        return 1;
    else if ( a->j < b->j )
        return -1;
    else
        return a->i - b->i;
}

int main()
{
    int i, j, k, neleC;
    double t;

    bzero(C, sizeof(int) * (N * N));
    bzero(C1, sizeof(int) * (N * N));
    bzero(C2, sizeof(int) * (N * N));

    for ( k = 0; k < ND; k++ )
    {
        AD[k].i = rand() % (N - 1);
        AD[k].j = rand() % (N - 1);
        AD[k].v = rand() % 100 + 1;
        while ( A[AD[k].i][AD[k].j] )
        {
            if ( AD[k].i < AD[k].j )
                AD[k].i = (AD[k].i + 1) % N;
            else
                AD[k].j = (AD[k].j + 1) % N;
        }
        A[AD[k].i][AD[k].j] = AD[k].v;
    }
    qsort(AD, ND, sizeof(tmd), cmp_fil); 

    for ( k = 0; k < ND; k++ )
    {
        BD[k].i = rand() % (N - 1);
        BD[k].j = rand() % (N - 1);
        BD[k].v = rand() % 100 + 1;
        while ( B[BD[k].i][BD[k].j] )
        {
            if ( BD[k].i < BD[k].j )
                BD[k].i = (BD[k].i + 1) % N;
            else
                BD[k].j = (BD[k].j + 1) % N;
        }
        B[BD[k].i][BD[k].j] = BD[k].v;
    }
    qsort(BD, ND, sizeof(tmd), cmp_col); 

    k = 0;
    for ( j = 0; j < N + 1; j++ )
    {
        while ( k < ND && j > BD[k].j )
            k++;
        jBD[j] = k;
    }

    neleC = 0;
    // ================================================================
    // VERSIÓN ACADÉMICA: 100% Operaciones + Privatización en Caché
    // ================================================================
    #pragma omp parallel
    {
        // Variables 100% locales que viven en la Caché ultrarrápida del procesador
        int VB_local[N];
        int VC_local[N];
        tmd CD_col[N];
        int count;

        // ---------------------------------------------------
        // 1. Matriu dispersa per matriu -> C1
        // ---------------------------------------------------
        #pragma omp for schedule(static) nowait
        for ( int i = 0; i < N; i++ )
        {
            // PREPARACIÓN: Copiamos los datos de RAM a Caché
            for ( int j = 0; j < N; j++ ) {
                VB_local[j] = B[j][i];
                VC_local[j] = 0;
            }

            // CÁLCULO: Fuerza bruta (640.000 ops sin saltarse nada). 
            // Al estar en Caché, esto vuela y no atasca a los demás hilos.
            for ( int k = 0; k < ND; k++ ) {
                VC_local[AD[k].i] += AD[k].v * VB_local[AD[k].j];
            }

            // VOLCADO: Escribimos en la lenta RAM solo 1 vez por columna
            for ( int j = 0; j < N; j++ ) {
                if (VC_local[j]) {
                    C1[j][i] = VC_local[j];
                }
            }
        }

        // ---------------------------------------------------
        // 2 y 3. Matriu dispersa per matriu dispersa -> C2 y CD
        // ---------------------------------------------------
        #pragma omp for schedule(static)
        for ( int i = 0; i < N; i++ )
        {
            // Limpieza local
            for ( int j = 0; j < N; j++ ) {
                VB_local[j] = 0;
                VC_local[j] = 0;
            }

            // PREPARACIÓN: Expandir columna en Caché
            for ( int k = jBD[i]; k < jBD[i + 1]; k++ ) {
                VB_local[BD[k].i] = BD[k].v;
            }

            // CÁLCULO: Fuerza bruta completa en Caché
            for ( int k = 0; k < ND; k++ ) {
                VC_local[AD[k].i] += AD[k].v * VB_local[AD[k].j];
            }

            count = 0;
            // VOLCADO Y COMPRESIÓN
            for ( int j = 0; j < N; j++ )
            {
                if ( VC_local[j] )
                {
                    C2[j][i] = VC_local[j]; // Matriz normal
                    
                    // Buffer disperso
                    CD_col[count].i = j;
                    CD_col[count].j = i;
                    CD_col[count].v = VC_local[j];
                    count++;
                }
            }

            // Atomic Capture para guardado global sin bloqueos masivos
            if (count > 0) {
                int start_idx;
                #pragma omp atomic capture
                {
                    start_idx = neleC;
                    neleC += count;
                }
                for ( int c = 0; c < count; c++ ) {
                    CD[start_idx + c] = CD_col[c];
                }
            }
        }
    } // FIN ZONA PARALELA


    // Comprovacio MD x M -> M i MD x MD -> M
    for ( i = 0; i < N; i++ )
        for ( j = 0; j < N; j++ )
            if ( C2[i][j] != C1[i][j] )
                printf("Diferencies C1 i C2 pos %d,%d: %d != %d\n", i, j, C1[i][j], C2[i][j]);

    // Comprovacio MD X MD -> M i MD x MD -> MD
    Suma = 0;
    for ( k = 0; k < neleC; k++ )
    {
        Suma += CD[k].v;
        if ( CD[k].v != C1[CD[k].i][CD[k].j] )
            printf("Diferencies C1 i CD a i:%d,j:%d,v%d, k:%d, vd:%d\n",
                   CD[k].i, CD[k].j, C1[CD[k].i][CD[k].j], k, CD[k].v);
    }

    printf("\nNumero elements de la matriu dispersa C %d\n", neleC);
    printf("Suma dels elements de C %lld \n", Suma);
    exit(0);
}