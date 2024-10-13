#ifndef MONO_ODOMETRY_H
#define MONO_ODOMETRY_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/viz.hpp>
#include <filesystem>
#include <string>
#include <exception>
#include <cmath>
#include <numeric>
#include <vector>

/**
 * @class MonoOdometry
 * @brief Classe para realizar odometria visual usando uma câmera monocular.
 *
 * Esta classe processa um vídeo, detecta características nas imagens e
 * estima a trajetória da câmera com base nas mudanças nas características
 * detectadas entre os frames do vídeo.
 */
class MonoOdometry
{
public:
    /**
     * @brief Construtor da classe MonoOdometry.
     *
     * @param video_folder Caminho para o arquivo de vídeo que será processado.
     */
    MonoOdometry(const std::string &video_folder);

    /**
     * @brief Método principal para processar o vídeo e realizar odometria.
     *
     * Este método lê o vídeo frame a frame, detecta características,
     * calcula a matriz essencial e a pose da câmera, e exibe os frames
     * processados.
     */
    void processVideos();

    /**
     * @brief Detects keypoints and computes descriptors from the current frame using ORB (Oriented FAST and Rotated BRIEF).
     *
     * This function takes a frame as input and uses the ORB feature detector to find keypoints and compute their corresponding
     * descriptors. The detected keypoints and descriptors are stored in the provided vectors.
     *
     * @param curr_frame The input image/frame from which keypoints and descriptors will be detected.
     * @param keypoints A vector to store the detected keypoints.
     * @param descriptors A matrix to store the computed descriptors corresponding to the keypoints.
     */
    void detectFeatures(const cv::Mat &curr_frame, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

    /**
     * @brief Calcula a distância entre as features atuais e as antigas, e os vetores de distância em x e y.
     *
     * Esta função calcula a distância euclidiana entre cada par de pontos correspondentes, bem como as diferenças
     * em x e y. No final, calcula a média das distâncias e dos vetores de distância em x e y.
     *
     * @param prev_keypoints Vetor de keypoints da imagem anterior.
     * @param curr_keypoints Vetor de keypoints da imagem atual.
     * @param good_matches Vetor de correspondências entre os keypoints das duas imagens.
     * @return std::tuple<double, double, double> Média da distância, média do vetor de distância em x, média do vetor de distância em y.
     */
    std::tuple<double, double, double> calculateDistances(const std::vector<cv::KeyPoint> &prev_keypoints, const std::vector<cv::KeyPoint> &curr_keypoints, const std::vector<cv::DMatch> &good_matches);

    /**
     * @brief Computa a odometria visual entre dois frames e atualiza a matriz de rotação e vetor de translação.
     *
     * @param prev_frame O frame anterior (em escala de cinza).
     * @param curr_frame O frame atual (em escala de cinza).
     */
    void computeOdometry(const std::vector<cv::Point2f> &prev_points, const std::vector<cv::Point2f> &curr_points, const cv::Mat &K);

    /**
     * @brief Realiza o rastreamento de características entre dois frames usando fluxo óptico.
     *
     * @param prev_frame O frame anterior (em escala de cinza).
     * @param curr_frame O frame atual (em escala de cinza).
     * @param prev_points Os Pontos anteriores.
     * @param curr_points Os Pontos atuais.
     */
    void trackFeatures(const cv::Mat &prev_frame, const cv::Mat &curr_frame, std::vector<cv::Point2f> &prev_points, std::vector<cv::Point2f> &curr_points);
    /**
     * @brief Desenha os pontos correspondentes e não correspondentes em um frame.
     *
     * Esta função marca os pontos correspondentes com um círculo verde e uma borda verde,
     * e os pontos não correspondentes com um círculo vermelho e uma borda vermelha.
     *
     * @param frame A imagem onde as features serão desenhadas (matriz de pixels).
     * @param matchedPoints Vetor de pontos correspondentes que serão desenhados em verde.
     * @param unmatchedPoints Vetor de pontos não correspondentes que serão desenhados em vermelho.
     */
    void drawFeatures(cv::Mat &frame, const std::vector<cv::Point2f> &matchedPoints, const std::vector<cv::Point2f> &unmatchedPoints);

    /**
     * @brief Calcula os ângulos de Euler (pitch, yaw, roll) a partir de uma matriz de rotação.
     *
     * Esta função extrai os ângulos de inclinação (pitch), guindagem (yaw) e rolamento (roll) a partir
     * de uma matriz de rotação 3x3. Os ângulos são retornados em radianos.
     *
     * @param R A matriz de rotação 3x3 que representa a orientação.
     * @param pitch Saída para o ângulo de inclinação (pitch) em radianos.
     * @param yaw Saída para o ângulo de guindagem (yaw) em radianos.
     * @param roll Saída para o ângulo de rolamento (roll) em radianos.
     */
    void calculateEulerAngles(const cv::Mat &R, double &pitch, double &yaw, double &roll);

private:
    std::string video_folder_;                // Caminho para o arquivo de vídeo.
    cv::Mat *frame;                           // Ponteiro para o frame atual do vídeo.
    std::vector<cv::Point3f> points_3d;       // Armazenar pontos 3D reconstruídos.
    cv::Mat R;                                // Matriz de rotação da câmera.
    cv::Mat t;                                // Vetor de translação da câmera.
    cv::Mat K;                                // Matriz de calibração da câmera.
    cv::Mat position;                         // Posição acumulada da câmera.
    std::vector<cv::Point2f> matchedPoints;   // Pontos correspondentes
    std::vector<cv::Point2f> unmatchedPoints; // Pontos não correspondentes
    std::vector<cv::KeyPoint> prev_keypoints; // Keypoints do frame anterior.
    cv::Mat prev_descriptors;                 // Descritores das características do frame anterior.
};

#endif // MONO_ODOMETRY_H
