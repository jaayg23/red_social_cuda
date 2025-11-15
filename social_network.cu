#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>

#define MAX_USERS 1000
#define MAX_POSTS 2000
#define MAX_TEXT_LEN 256
#define MAX_HASHTAG_LEN 32
#define THREADS_PER_BLOCK 256

// ============================================================================
// ESTRUCTURAS DE DATOS
// ============================================================================

// Tipos de usuario
enum UserType { PERSON = 0, COMPANY = 1 };

// Tipo de interacción con publicación
enum Interaction { NONE = 0, LIKE = 1, DISLIKE = 2 };

// Estructura para personas (SoA - Structure of Arrays)
struct Persons {
    int count;
    int ids[MAX_USERS];
    char names[MAX_USERS][64];
};

// Estructura para empresas
struct Companies {
    int count;
    int ids[MAX_USERS];
    char names[MAX_USERS][64];
};

// Estructura para publicaciones
struct Posts {
    int count;
    int ids[MAX_POSTS];
    char texts[MAX_POSTS][MAX_TEXT_LEN];
    char hashtags[MAX_POSTS][MAX_HASHTAG_LEN];
    int author_ids[MAX_POSTS];
    UserType author_types[MAX_POSTS];
    int original_post_id[MAX_POSTS];  // -1 si es original, sino ID del post original
};

// Relaciones entre usuarios (matrices de adyacencia)
// 0 = no relación, 1 = relación
struct Relations {
    // Persona -> Persona
    int person_follows_person[MAX_USERS][MAX_USERS];
    int person_blocks_person[MAX_USERS][MAX_USERS];

    // Persona -> Empresa
    int person_follows_company[MAX_USERS][MAX_USERS];
    int person_is_client[MAX_USERS][MAX_USERS];
    int person_works_at[MAX_USERS][MAX_USERS];
    int person_blocked_by_company[MAX_USERS][MAX_USERS];

    // Empresa -> Empresa
    int company_follows_company[MAX_USERS][MAX_USERS];
    int company_recommends_company[MAX_USERS][MAX_USERS];
    int company_blocks_company[MAX_USERS][MAX_USERS];

    // Empresa -> Persona
    int company_blocks_person[MAX_USERS][MAX_USERS];
};

// Interacciones con publicaciones
struct PostInteractions {
    Interaction person_interactions[MAX_USERS][MAX_POSTS];
    Interaction company_interactions[MAX_USERS][MAX_POSTS];
};

// ============================================================================
// KERNELS CUDA - OPERACIONES BÁSICAS
// ============================================================================

// Kernel para contar seguidores de una persona
__global__ void count_person_followers_kernel(int person_idx, int* relations, int num_persons, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int local_count[THREADS_PER_BLOCK];
    local_count[threadIdx.x] = 0;

    if (tid < num_persons) {
        if (relations[tid * MAX_USERS + person_idx] == 1) {
            local_count[threadIdx.x] = 1;
        }
    }

    __syncthreads();

    // Reducción paralela
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            local_count[threadIdx.x] += local_count[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, local_count[0]);
    }
}

// Kernel para contar seguidores de una empresa
__global__ void count_company_followers_kernel(int company_idx,
                                               int* person_follows_company,
                                               int* company_follows_company,
                                               int num_persons,
                                               int num_companies,
                                               int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int local_count[THREADS_PER_BLOCK];
    local_count[threadIdx.x] = 0;

    // Contar personas que siguen a la empresa
    if (tid < num_persons) {
        if (person_follows_company[tid * MAX_USERS + company_idx] == 1) {
            local_count[threadIdx.x] = 1;
        }
    }
    // Contar empresas que siguen a la empresa
    else if (tid < num_persons + num_companies) {
        int company_tid = tid - num_persons;
        if (company_follows_company[company_tid * MAX_USERS + company_idx] == 1) {
            local_count[threadIdx.x] = 1;
        }
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            local_count[threadIdx.x] += local_count[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, local_count[0]);
    }
}

// Kernel para contar likes de una publicación
__global__ void count_post_likes_kernel(int post_idx,
                                        Interaction* person_interactions,
                                        Interaction* company_interactions,
                                        int num_persons,
                                        int num_companies,
                                        int* likes,
                                        int* dislikes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int local_likes[THREADS_PER_BLOCK];
    __shared__ int local_dislikes[THREADS_PER_BLOCK];
    local_likes[threadIdx.x] = 0;
    local_dislikes[threadIdx.x] = 0;

    if (tid < num_persons) {
        Interaction inter = person_interactions[tid * MAX_POSTS + post_idx];
        if (inter == LIKE) local_likes[threadIdx.x] = 1;
        else if (inter == DISLIKE) local_dislikes[threadIdx.x] = 1;
    }
    else if (tid < num_persons + num_companies) {
        int company_tid = tid - num_persons;
        Interaction inter = company_interactions[company_tid * MAX_POSTS + post_idx];
        if (inter == LIKE) local_likes[threadIdx.x] = 1;
        else if (inter == DISLIKE) local_dislikes[threadIdx.x] = 1;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            local_likes[threadIdx.x] += local_likes[threadIdx.x + stride];
            local_dislikes[threadIdx.x] += local_dislikes[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(likes, local_likes[0]);
        atomicAdd(dislikes, local_dislikes[0]);
    }
}

// Kernel para encontrar publicaciones con un hashtag
__global__ void find_posts_by_hashtag_kernel(char (*hashtags)[MAX_HASHTAG_LEN],
                                             int num_posts,
                                             const char* target_hashtag,
                                             int* result_indices,
                                             int* result_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_posts) {
        bool match = true;
        for (int i = 0; i < MAX_HASHTAG_LEN; i++) {
            if (hashtags[tid][i] != target_hashtag[i]) {
                match = false;
                break;
            }
            if (hashtags[tid][i] == '\0') break;
        }

        if (match && hashtags[tid][0] != '\0') {
            int idx = atomicAdd(result_count, 1);
            if (idx < MAX_POSTS) {
                result_indices[idx] = tid;
            }
        }
    }
}

// Kernel para verificar si una persona puede ver una publicación
__global__ void check_visibility_kernel(int post_idx,
                                       int author_id,
                                       int* person_follows_person,
                                       int* person_blocks_person,
                                       int num_persons,
                                       int* can_view) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_persons) {
        // El autor siempre puede ver su propia publicación
        if (tid == author_id) {
            can_view[tid] = 1;
            return;
        }

        bool is_follower = (person_follows_person[tid * MAX_USERS + author_id] == 1);
        bool is_blocked = (person_blocks_person[author_id * MAX_USERS + tid] == 1);

        // Puede ver si es seguidor y no está bloqueado
        if (is_follower && !is_blocked) {
            can_view[tid] = 1;
            return;
        }

        // Buscar si es seguidor de un seguidor (grado 2)
        for (int i = 0; i < num_persons; i++) {
            bool i_follows_author = (person_follows_person[i * MAX_USERS + author_id] == 1);
            bool tid_follows_i = (person_follows_person[tid * MAX_USERS + i] == 1);
            bool i_blocked_by_author = (person_blocks_person[author_id * MAX_USERS + i] == 1);

            if (i_follows_author && !i_blocked_by_author && tid_follows_i) {
                can_view[tid] = 1;
                return;
            }
        }

        can_view[tid] = 0;
    }
}

// ============================================================================
// FUNCIONES HOST
// ============================================================================

// Inicializar datos de ejemplo
void initialize_sample_data(Persons* persons, Companies* companies, Posts* posts,
                           Relations* relations, PostInteractions* interactions) {
    // Personas
    persons->count = 6;
    int person_ids[] = {0, 1, 2, 3, 4, 5};
    const char* person_names[] = {"Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"};
    for (int i = 0; i < persons->count; i++) {
        persons->ids[i] = person_ids[i];
        strcpy(persons->names[i], person_names[i]);
    }

    // Empresas
    companies->count = 3;
    int company_ids[] = {0, 1, 2};
    const char* company_names[] = {"TechCorp", "SocialHub", "DataInc"};
    for (int i = 0; i < companies->count; i++) {
        companies->ids[i] = company_ids[i];
        strcpy(companies->names[i], company_names[i]);
    }

    // Inicializar todas las relaciones a 0
    memset(relations, 0, sizeof(Relations));

    // Relaciones persona-persona (seguimiento)
    relations->person_follows_person[0][1] = 1; // Alice sigue a Bob
    relations->person_follows_person[1][0] = 1; // Bob sigue a Alice (amigos)
    relations->person_follows_person[2][1] = 1; // Charlie sigue a Bob
    relations->person_follows_person[3][0] = 1; // Diana sigue a Alice
    relations->person_follows_person[4][2] = 1; // Eve sigue a Charlie
    relations->person_follows_person[5][3] = 1; // Frank sigue a Diana

    // Bloqueos
    relations->person_blocks_person[1][2] = 1; // Bob bloquea a Charlie

    // Relaciones persona-empresa
    relations->person_follows_company[0][0] = 1; // Alice sigue a TechCorp
    relations->person_follows_company[1][0] = 1; // Bob sigue a TechCorp
    relations->person_follows_company[2][1] = 1; // Charlie sigue a SocialHub
    relations->person_is_client[0][0] = 1;       // Alice es cliente de TechCorp
    relations->person_is_client[3][0] = 1;       // Diana es cliente de TechCorp
    relations->person_works_at[4][1] = 1;        // Eve trabaja en SocialHub

    // Relaciones empresa-empresa
    relations->company_follows_company[0][1] = 1;    // TechCorp sigue a SocialHub
    relations->company_recommends_company[0][2] = 1; // TechCorp recomienda a DataInc
    relations->company_recommends_company[1][2] = 1; // SocialHub recomienda a DataInc

    // Publicaciones
    posts->count = 10;

    // Publicaciones de personas
    posts->ids[0] = 0;
    strcpy(posts->texts[0], "Hola mundo! #tech");
    strcpy(posts->hashtags[0], "#tech");
    posts->author_ids[0] = 0;
    posts->author_types[0] = PERSON;
    posts->original_post_id[0] = -1;

    posts->ids[1] = 1;
    strcpy(posts->texts[1], "Me encanta programar #coding");
    strcpy(posts->hashtags[1], "#coding");
    posts->author_ids[1] = 1;
    posts->author_types[1] = PERSON;
    posts->original_post_id[1] = -1;

    posts->ids[2] = 2;
    strcpy(posts->texts[2], "Hermoso dia! #life");
    strcpy(posts->hashtags[2], "#life");
    posts->author_ids[2] = 2;
    posts->author_types[2] = PERSON;
    posts->original_post_id[2] = -1;

    posts->ids[3] = 3;
    strcpy(posts->texts[3], "CUDA es increible #tech");
    strcpy(posts->hashtags[3], "#tech");
    posts->author_ids[3] = 0;
    posts->author_types[3] = PERSON;
    posts->original_post_id[3] = -1;

    // Republicación: Bob republica post de Alice
    posts->ids[4] = 4;
    strcpy(posts->texts[4], "Hola mundo! #tech");
    strcpy(posts->hashtags[4], "#tech");
    posts->author_ids[4] = 1;
    posts->author_types[4] = PERSON;
    posts->original_post_id[4] = 0;

    // Publicaciones de empresas
    posts->ids[5] = 5;
    strcpy(posts->texts[5], "Nuevos productos disponibles #tech");
    strcpy(posts->hashtags[5], "#tech");
    posts->author_ids[5] = 0;
    posts->author_types[5] = COMPANY;
    posts->original_post_id[5] = -1;

    posts->ids[6] = 6;
    strcpy(posts->texts[6], "Unete a nuestra red #social");
    strcpy(posts->hashtags[6], "#social");
    posts->author_ids[6] = 1;
    posts->author_types[6] = COMPANY;
    posts->original_post_id[6] = -1;

    posts->ids[7] = 7;
    strcpy(posts->texts[7], "Analiza tus datos #data");
    strcpy(posts->hashtags[7], "#data");
    posts->author_ids[7] = 2;
    posts->author_types[7] = COMPANY;
    posts->original_post_id[7] = -1;

    // Empresa recomienda publicación de otra empresa
    posts->ids[8] = 8;
    strcpy(posts->texts[8], "Analiza tus datos #data");
    strcpy(posts->hashtags[8], "#data");
    posts->author_ids[8] = 0;
    posts->author_types[8] = COMPANY;
    posts->original_post_id[8] = 7;

    posts->ids[9] = 9;
    strcpy(posts->texts[9], "Gran evento de tecnologia #tech");
    strcpy(posts->hashtags[9], "#tech");
    posts->author_ids[9] = 0;
    posts->author_types[9] = COMPANY;
    posts->original_post_id[9] = -1;

    // Interacciones con publicaciones
    memset(interactions, 0, sizeof(PostInteractions));

    // Likes
    interactions->person_interactions[0][1] = LIKE;  // Alice le gusta post 1 de Bob
    interactions->person_interactions[1][0] = LIKE;  // Bob le gusta post 0 de Alice
    interactions->person_interactions[2][1] = LIKE;  // Charlie le gusta post 1 de Bob
    interactions->person_interactions[0][5] = LIKE;  // Alice le gusta post 5 de TechCorp
    interactions->person_interactions[1][5] = LIKE;  // Bob le gusta post 5 de TechCorp
    interactions->person_interactions[3][5] = LIKE;  // Diana le gusta post 5 de TechCorp
    interactions->person_interactions[0][3] = LIKE;  // Alice le gusta su propio post 3

    // Dislikes
    interactions->person_interactions[4][2] = DISLIKE;  // Eve no le gusta post 2
    interactions->person_interactions[2][6] = DISLIKE;  // Charlie no le gusta post 6

    // Likes de empresas
    interactions->company_interactions[0][6] = LIKE;  // TechCorp le gusta post 6 de SocialHub
}

// Contar seguidores de una persona
int count_person_followers(int person_idx, Relations* relations, int num_persons) {
    int *d_relations, *d_result;
    int h_result = 0;

    cudaMalloc(&d_relations, MAX_USERS * MAX_USERS * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_relations, relations->person_follows_person,
               MAX_USERS * MAX_USERS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (num_persons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    count_person_followers_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        person_idx, d_relations, num_persons, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_relations);
    cudaFree(d_result);

    return h_result;
}

// Contar seguidores de una empresa
int count_company_followers(int company_idx, Relations* relations,
                            int num_persons, int num_companies) {
    int *d_person_follows, *d_company_follows, *d_result;
    int h_result = 0;

    cudaMalloc(&d_person_follows, MAX_USERS * MAX_USERS * sizeof(int));
    cudaMalloc(&d_company_follows, MAX_USERS * MAX_USERS * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_person_follows, relations->person_follows_company,
               MAX_USERS * MAX_USERS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_company_follows, relations->company_follows_company,
               MAX_USERS * MAX_USERS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    int total_users = num_persons + num_companies;
    int num_blocks = (total_users + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    count_company_followers_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        company_idx, d_person_follows, d_company_follows,
        num_persons, num_companies, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_person_follows);
    cudaFree(d_company_follows);
    cudaFree(d_result);

    return h_result;
}

// Contar likes y dislikes de una publicación
void count_post_reactions(int post_idx, PostInteractions* interactions,
                         int num_persons, int num_companies,
                         int* likes, int* dislikes) {
    Interaction *d_person_inter, *d_company_inter;
    int *d_likes, *d_dislikes;
    int h_likes = 0, h_dislikes = 0;

    cudaMalloc(&d_person_inter, MAX_USERS * MAX_POSTS * sizeof(Interaction));
    cudaMalloc(&d_company_inter, MAX_USERS * MAX_POSTS * sizeof(Interaction));
    cudaMalloc(&d_likes, sizeof(int));
    cudaMalloc(&d_dislikes, sizeof(int));

    cudaMemcpy(d_person_inter, interactions->person_interactions,
               MAX_USERS * MAX_POSTS * sizeof(Interaction), cudaMemcpyHostToDevice);
    cudaMemcpy(d_company_inter, interactions->company_interactions,
               MAX_USERS * MAX_POSTS * sizeof(Interaction), cudaMemcpyHostToDevice);
    cudaMemcpy(d_likes, &h_likes, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dislikes, &h_dislikes, sizeof(int), cudaMemcpyHostToDevice);

    int total_users = num_persons + num_companies;
    int num_blocks = (total_users + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    count_post_likes_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        post_idx, d_person_inter, d_company_inter,
        num_persons, num_companies, d_likes, d_dislikes);

    cudaMemcpy(&h_likes, d_likes, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_dislikes, d_dislikes, sizeof(int), cudaMemcpyDeviceToHost);

    *likes = h_likes;
    *dislikes = h_dislikes;

    cudaFree(d_person_inter);
    cudaFree(d_company_inter);
    cudaFree(d_likes);
    cudaFree(d_dislikes);
}

// ============================================================================
// QUERIES PRINCIPALES
// ============================================================================

void query_followers(Persons* persons, Companies* companies, Relations* relations) {
    printf("\n========== CANTIDAD DE SEGUIDORES ==========\n");

    printf("\n--- Personas ---\n");
    for (int i = 0; i < persons->count; i++) {
        int followers = count_person_followers(i, relations, persons->count);
        printf("%s: %d seguidores\n", persons->names[i], followers);
    }

    printf("\n--- Empresas ---\n");
    for (int i = 0; i < companies->count; i++) {
        int followers = count_company_followers(i, relations,
                                                persons->count, companies->count);
        printf("%s: %d seguidores\n", companies->names[i], followers);
    }
}

void query_post_reactions(Posts* posts, PostInteractions* interactions,
                         int num_persons, int num_companies) {
    printf("\n========== REACCIONES POR PUBLICACION ==========\n");

    for (int i = 0; i < posts->count; i++) {
        int likes, dislikes;
        count_post_reactions(i, interactions, num_persons, num_companies,
                           &likes, &dislikes);

        printf("\nPost %d: \"%s\"\n", posts->ids[i], posts->texts[i]);
        printf("  Likes: %d | Dislikes: %d\n", likes, dislikes);
    }
}

void query_top_posts(Posts* posts, PostInteractions* interactions,
                    int num_persons, int num_companies) {
    printf("\n========== TOP 5 PUBLICACIONES ==========\n");

    // Calcular likes para todos los posts
    int likes[MAX_POSTS];
    for (int i = 0; i < posts->count; i++) {
        int dislikes;
        count_post_reactions(i, interactions, num_persons, num_companies,
                           &likes[i], &dislikes);
    }

    // Crear array de índices y ordenar
    int indices[MAX_POSTS];
    for (int i = 0; i < posts->count; i++) indices[i] = i;

    // Ordenar por likes (descendente)
    for (int i = 0; i < posts->count - 1; i++) {
        for (int j = i + 1; j < posts->count; j++) {
            if (likes[indices[i]] < likes[indices[j]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }

    printf("\n--- Top 5 con MAS likes ---\n");
    for (int i = 0; i < 5 && i < posts->count; i++) {
        int idx = indices[i];
        printf("%d. \"%s\" - %d likes\n", i+1, posts->texts[idx], likes[idx]);
    }

    printf("\n--- Top 5 con MENOS likes ---\n");
    for (int i = 0; i < 5 && i < posts->count; i++) {
        int idx = indices[posts->count - 1 - i];
        printf("%d. \"%s\" - %d likes\n", i+1, posts->texts[idx], likes[idx]);
    }
}

void query_blocked_followers(Persons* persons, Companies* companies, Relations* relations) {
    printf("\n========== SEGUIDORES BLOQUEADOS ==========\n");

    printf("\n--- Personas que han bloqueado seguidores ---\n");
    for (int i = 0; i < persons->count; i++) {
        bool has_blocks = false;
        for (int j = 0; j < persons->count; j++) {
            if (relations->person_blocks_person[i][j] == 1) {
                if (!has_blocks) {
                    printf("%s ha bloqueado a:\n", persons->names[i]);
                    has_blocks = true;
                }
                printf("  - %s\n", persons->names[j]);
            }
        }
    }

    printf("\n--- Empresas que han bloqueado seguidores ---\n");
    for (int i = 0; i < companies->count; i++) {
        bool has_blocks = false;
        for (int j = 0; j < persons->count; j++) {
            if (relations->company_blocks_person[i][j] == 1) {
                if (!has_blocks) {
                    printf("%s ha bloqueado a:\n", companies->names[i]);
                    has_blocks = true;
                }
                printf("  - %s (persona)\n", persons->names[j]);
            }
        }
    }
}

void query_company_recommendations(Companies* companies, Relations* relations) {
    printf("\n========== RECOMENDACIONES DE EMPRESAS ==========\n");

    for (int i = 0; i < companies->count; i++) {
        int rec_count = 0;
        printf("\n%s recibio recomendaciones de:\n", companies->names[i]);

        for (int j = 0; j < companies->count; j++) {
            if (relations->company_recommends_company[j][i] == 1) {
                printf("  - %s\n", companies->names[j]);
                rec_count++;
            }
        }

        if (rec_count == 0) {
            printf("  (ninguna)\n");
        }
    }
}

void query_hashtags(Posts* posts) {
    printf("\n========== ANALISIS DE HASHTAGS ==========\n");

    // Contar hashtags
    char unique_hashtags[MAX_POSTS][MAX_HASHTAG_LEN];
    int hashtag_counts[MAX_POSTS];
    int num_unique = 0;

    for (int i = 0; i < posts->count; i++) {
        if (posts->hashtags[i][0] == '\0') continue;

        bool found = false;
        for (int j = 0; j < num_unique; j++) {
            if (strcmp(unique_hashtags[j], posts->hashtags[i]) == 0) {
                hashtag_counts[j]++;
                found = true;
                break;
            }
        }

        if (!found) {
            strcpy(unique_hashtags[num_unique], posts->hashtags[i]);
            hashtag_counts[num_unique] = 1;
            num_unique++;
        }
    }

    // Encontrar más popular
    int max_idx = 0;
    for (int i = 1; i < num_unique; i++) {
        if (hashtag_counts[i] > hashtag_counts[max_idx]) {
            max_idx = i;
        }
    }

    printf("\nHashtag mas usado: %s (%d publicaciones)\n",
           unique_hashtags[max_idx], hashtag_counts[max_idx]);

    printf("\nTodos los hashtags:\n");
    for (int i = 0; i < num_unique; i++) {
        printf("  %s: %d publicaciones\n", unique_hashtags[i], hashtag_counts[i]);
    }
}

void query_best_customers(Persons* persons, Companies* companies,
                         Relations* relations, PostInteractions* interactions,
                         Posts* posts) {
    printf("\n========== MEJORES CLIENTES DE EMPRESAS ==========\n");

    for (int c = 0; c < companies->count; c++) {
        printf("\n%s - Clientes que mas gustan de sus publicaciones:\n",
               companies->names[c]);

        int customer_likes[MAX_USERS] = {0};

        // Contar likes de clientes a publicaciones de esta empresa
        for (int p = 0; p < persons->count; p++) {
            if (relations->person_is_client[p][c] == 0) continue;

            for (int post_i = 0; post_i < posts->count; post_i++) {
                if (posts->author_types[post_i] == COMPANY &&
                    posts->author_ids[post_i] == c) {
                    if (interactions->person_interactions[p][post_i] == LIKE) {
                        customer_likes[p]++;
                    }
                }
            }
        }

        // Mostrar clientes ordenados por likes
        bool found_any = false;
        for (int likes = 10; likes >= 0; likes--) {
            for (int p = 0; p < persons->count; p++) {
                if (customer_likes[p] == likes && likes > 0) {
                    printf("  - %s: %d likes\n", persons->names[p], likes);
                    found_any = true;
                }
            }
        }

        if (!found_any) {
            printf("  (no hay clientes con likes)\n");
        }
    }
}

void query_top_companies_by_likes(Companies* companies, Posts* posts,
                                 PostInteractions* interactions,
                                 int num_persons, int num_companies) {
    printf("\n========== EMPRESAS CON MAS/MENOS LIKES ==========\n");

    int company_likes[MAX_USERS] = {0};
    int company_dislikes[MAX_USERS] = {0};

    for (int i = 0; i < posts->count; i++) {
        if (posts->author_types[i] == COMPANY) {
            int likes, dislikes;
            count_post_reactions(i, interactions, num_persons, num_companies,
                               &likes, &dislikes);
            company_likes[posts->author_ids[i]] += likes;
            company_dislikes[posts->author_ids[i]] += dislikes;
        }
    }

    printf("\n--- Empresas con MAS likes ---\n");
    for (int i = 0; i < num_companies; i++) {
        printf("%s: %d likes totales\n", companies->names[i], company_likes[i]);
    }

    printf("\n--- Empresas con MAS dislikes ---\n");
    for (int i = 0; i < num_companies; i++) {
        printf("%s: %d dislikes totales\n", companies->names[i], company_dislikes[i]);
    }
}

void query_top_companies_by_recommendations(Companies* companies, Relations* relations) {
    printf("\n========== EMPRESAS CON MAS RECOMENDACIONES ==========\n");

    int rec_counts[MAX_USERS];
    for (int i = 0; i < companies->count; i++) {
        rec_counts[i] = 0;
        for (int j = 0; j < companies->count; j++) {
            if (relations->company_recommends_company[j][i] == 1) {
                rec_counts[i]++;
            }
        }
    }

    // Crear índices y ordenar
    int indices[MAX_USERS];
    for (int i = 0; i < companies->count; i++) indices[i] = i;

    for (int i = 0; i < companies->count - 1; i++) {
        for (int j = i + 1; j < companies->count; j++) {
            if (rec_counts[indices[i]] < rec_counts[indices[j]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }

    for (int i = 0; i < companies->count; i++) {
        int idx = indices[i];
        printf("%d. %s: %d recomendaciones\n", i+1, companies->names[idx], rec_counts[idx]);
    }
}

void query_visibility_of_post(int post_idx, Posts* posts, Persons* persons,
                              Relations* relations) {
    if (posts->author_types[post_idx] == COMPANY) {
        printf("\n========== VISIBILIDAD DEL POST %d (EMPRESA) ==========\n", post_idx);
        printf("Post: \"%s\"\n", posts->texts[post_idx]);
        printf("Autor: Empresa %s\n\n", "");
        printf("Todos los usuarios pueden ver esta publicacion (es de una empresa)\n");
        return;
    }

    printf("\n========== VISIBILIDAD DEL POST %d (PERSONA) ==========\n", post_idx);
    printf("Post: \"%s\"\n", posts->texts[post_idx]);
    printf("Autor: %s\n\n", persons->names[posts->author_ids[post_idx]]);

    int author_id = posts->author_ids[post_idx];
    int *d_relations, *d_blocks, *d_can_view;
    int h_can_view[MAX_USERS] = {0};

    cudaMalloc(&d_relations, MAX_USERS * MAX_USERS * sizeof(int));
    cudaMalloc(&d_blocks, MAX_USERS * MAX_USERS * sizeof(int));
    cudaMalloc(&d_can_view, MAX_USERS * sizeof(int));

    cudaMemcpy(d_relations, relations->person_follows_person,
               MAX_USERS * MAX_USERS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blocks, relations->person_blocks_person,
               MAX_USERS * MAX_USERS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_can_view, h_can_view, MAX_USERS * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (persons->count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    check_visibility_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        post_idx, author_id, d_relations, d_blocks, persons->count, d_can_view);

    cudaMemcpy(h_can_view, d_can_view, MAX_USERS * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Personas que pueden ver esta publicacion:\n");
    for (int i = 0; i < persons->count; i++) {
        if (h_can_view[i] == 1) {
            printf("  - %s\n", persons->names[i]);
        }
    }

    cudaFree(d_relations);
    cudaFree(d_blocks);
    cudaFree(d_can_view);
}

void query_influence_network(int person_idx, int degree, Persons* persons,
                            Relations* relations) {
    printf("\n========== RED DE INFLUENCIA: %s (grado %d) ==========\n",
           persons->names[person_idx], degree);

    bool visited[MAX_USERS] = {false};
    int current_level[MAX_USERS], next_level[MAX_USERS];
    int current_size = 1, next_size = 0;

    current_level[0] = person_idx;
    visited[person_idx] = true;

    for (int d = 0; d < degree; d++) {
        printf("\n--- Grado %d ---\n", d + 1);
        next_size = 0;

        for (int i = 0; i < current_size; i++) {
            int current_person = current_level[i];

            for (int j = 0; j < persons->count; j++) {
                if (relations->person_follows_person[j][current_person] == 1 && !visited[j]) {
                    printf("  %s\n", persons->names[j]);
                    visited[j] = true;
                    next_level[next_size++] = j;
                }
            }
        }

        if (next_size == 0) {
            printf("  (no hay mas seguidores en este grado)\n");
            break;
        }

        memcpy(current_level, next_level, next_size * sizeof(int));
        current_size = next_size;
    }
}

void query_users_by_hashtag(const char* hashtag, Posts* posts, Persons* persons,
                            Companies* companies) {
    printf("\n========== USUARIOS QUE PUBLICARON %s ==========\n", hashtag);

    bool found_persons[MAX_USERS] = {false};
    bool found_companies[MAX_USERS] = {false};

    for (int i = 0; i < posts->count; i++) {
        if (strcmp(posts->hashtags[i], hashtag) == 0) {
            if (posts->author_types[i] == PERSON) {
                found_persons[posts->author_ids[i]] = true;
            } else {
                found_companies[posts->author_ids[i]] = true;
            }
        }
    }

    printf("\nPersonas:\n");
    bool any_person = false;
    for (int i = 0; i < persons->count; i++) {
        if (found_persons[i]) {
            printf("  - %s\n", persons->names[i]);
            any_person = true;
        }
    }
    if (!any_person) printf("  (ninguna)\n");

    printf("\nEmpresas:\n");
    bool any_company = false;
    for (int i = 0; i < companies->count; i++) {
        if (found_companies[i]) {
            printf("  - %s\n", companies->names[i]);
            any_company = true;
        }
    }
    if (!any_company) printf("  (ninguna)\n");
}

void query_posts_by_hashtag(const char* hashtag, Posts* posts) {
    printf("\n========== PUBLICACIONES CON %s ==========\n", hashtag);

    bool found_any = false;
    for (int i = 0; i < posts->count; i++) {
        if (strcmp(posts->hashtags[i], hashtag) == 0) {
            printf("  Post %d: \"%s\"\n", posts->ids[i], posts->texts[i]);
            found_any = true;
        }
    }

    if (!found_any) {
        printf("  (no se encontraron publicaciones)\n");
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("========================================\n");
    printf("  RED SOCIAL CON CUDA\n");
    printf("========================================\n");

    // Inicializar estructuras en host
    Persons* persons = new Persons();
    Companies* companies = new Companies();
    Posts* posts = new Posts();
    Relations* relations = new Relations();
    PostInteractions* interactions = new PostInteractions();

    // Cargar datos de ejemplo
    initialize_sample_data(persons, companies, posts, relations, interactions);

    printf("\nDatos cargados:\n");
    printf("  - %d personas\n", persons->count);
    printf("  - %d empresas\n", companies->count);
    printf("  - %d publicaciones\n", posts->count);

    // Ejecutar queries
    query_followers(persons, companies, relations);
    query_post_reactions(posts, interactions, persons->count, companies->count);
    query_top_posts(posts, interactions, persons->count, companies->count);
    query_blocked_followers(persons, companies, relations);
    query_company_recommendations(companies, relations);
    query_top_companies_by_recommendations(companies, relations);
    query_hashtags(posts);
    query_posts_by_hashtag("#tech", posts);
    query_users_by_hashtag("#tech", posts, persons, companies);
    query_best_customers(persons, companies, relations, interactions, posts);
    query_top_companies_by_likes(companies, posts, interactions,
                                persons->count, companies->count);

    // Ejemplos de visibilidad y red de influencia
    query_visibility_of_post(0, posts, persons, relations);  // Post de Alice
    query_visibility_of_post(5, posts, persons, relations);  // Post de empresa
    query_influence_network(0, 2, persons, relations);       // Red de Alice (grado 2)
    query_influence_network(1, 2, persons, relations);       // Red de Bob (grado 2)

    printf("\n========================================\n");
    printf("  FIN DE CONSULTAS\n");
    printf("========================================\n");

    // Limpiar
    delete persons;
    delete companies;
    delete posts;
    delete relations;
    delete interactions;

    return 0;
}
