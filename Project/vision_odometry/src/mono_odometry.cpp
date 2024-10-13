#include "mono_odometry.h"

/**
 * @brief Construtor da classe MonoOdometry.
 *
 * Inicializa o caminho do vídeo, cria um novo frame e
 * inicializa as matrizes de rotação e translação.
 *
 * @param video_folder Caminho para o arquivo de vídeo que será processado.
 */
MonoOdometry::MonoOdometry(const std::string &video_folder)
{
    video_folder_ = video_folder;
    frame = new cv::Mat();
    R = cv::Mat::eye(3, 3, CV_64F);          // Inicializa R como matriz identidade
    t = cv::Mat::zeros(3, 1, CV_64F);        // Inicializa t como vetor zero
    position = cv::Mat::zeros(3, 1, CV_64F); // Inicializa a posição acumulada como zeros

    // Inicializa a matriz de calibração (K) com valores padrão (ajuste conforme necessário)
    K = (cv::Mat_<double>(3, 3) << 924.2758995009278, 0.0, 640.5,
         0.0, 924.2759313476419, 360.5,
         0.0, 0.0, 1.0);
}
/**
 * @brief Processa o vídeo para realizar odometria visual.
 *
 * Este método lê o vídeo frame a frame, detecta características usando
 * o algoritmo ORB, calcula a matriz essencial entre frames, recupera a
 * pose da câmera e exibe os frames processados em uma janela.
 */
void MonoOdometry::processVideos()
{
    try
    {
        cv::VideoCapture cap(video_folder_);
        if (!cap.isOpened())
        {
            std::cout << "Cannot open the video file. \n";
            return;
        }

        std::vector<cv::Point2f> prev_points; // Pontos do frame anterior.
        std::vector<cv::Point2f> curr_points; // Pontos do frame atual.

        cv::Mat descriptors;                 // Descritores das características.
        std::vector<cv::KeyPoint> keypoints; // Keypoints detectados.

        cv::Mat prev_frame, curr_frame;

        while (true)
        {
            // Lê o próximo frame do vídeo
            if (!cap.read(*frame))
            {
                std::cout << "\nCannot read the video file.\n";
                break;
            }

            // Verifica se o frame é válido
            if (frame->empty())
            {
                std::cout << "Empty frame encountered. Skipping...\n";
                continue;
            }

            // Converte o frame atual para escala de cinza
            cv::cvtColor(*frame, curr_frame, cv::COLOR_BGR2GRAY);

            trackFeatures(prev_frame, curr_frame, prev_points, curr_points); // Rastreamento de características

            // Chama as funções de processamento de odometria
            if (!prev_points.empty())
            {
                computeOdometry(prev_points, curr_points, K);         // Computa a odometria visual
                drawFeatures(*frame, matchedPoints, unmatchedPoints); // Desenha os pontos correspondentes e não correspondentes
            }

            // Armazena os pontos atuais e o frame para o próximo loop
            prev_points.clear();
            cv::KeyPoint::convert(keypoints, prev_points);
            prev_frame = curr_frame.clone();

            // Exibe o frame com os resultados
            cv::imshow("Mono Odometry", *frame);
            char key = (char)cv::waitKey(30);
            if (key == 27) // Pressione 'ESC' para sair
                break;
        }
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "OpenCV exception: " << e.what() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred!" << std::endl;
    }
}

/**
 * @brief Detects keypoints and computes descriptors from the current frame using ORB (Oriented FAST and Rotated BRIEF).
 *
 * This function takes a frame as input and uses the ORB feature detector to find keypoints and compute their corresponding
 * descriptors. The detected keypoints and descriptors are stored in the provided vectors.
 *
 * @param curr_frame The input image/frame from which keypoints and descriptors will be detected.
 * @param keypoints            // Detecta características no frame atual
            detectFeatures(curr_frame, keypoints, descriptors);A vector to store the detected keypoints.
 * @param descriptors A matrix to store the computed descriptors corresponding to the keypoints.
 */
void MonoOdometry::detectFeatures(const cv::Mat &curr_frame, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    // Create a SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Detect keypoints and compute descriptors
    sift->detectAndCompute(curr_frame, cv::noArray(), keypoints, descriptors);
}

/**
 * @brief Realiza o rastreamento de características entre dois frames usando fluxo óptico.
 *
 * @param prev_frame O frame anterior (em escala de cinza).
 * @param curr_frame O frame atual (em escala de cinza).
 */
void MonoOdometry::trackFeatures(const cv::Mat &prev_frame, const cv::Mat &curr_frame, std::vector<cv::Point2f> &prev_points, std::vector<cv::Point2f> &curr_points)
{
    std::vector<cv::KeyPoint> curr_keypoints;
    cv::Mat curr_descriptors;

    // Detectar keypoints e descritores no frame atual
    detectFeatures(curr_frame, curr_keypoints, curr_descriptors);

    if (curr_keypoints.empty() || curr_descriptors.empty())
    {
        std::cout << "Nenhum keypoint encontrado no frame atual.\n";
        return;
    }

    if (prev_keypoints.empty() || prev_descriptors.empty())
    {
        // Atualizar os keypoints e descritores para o próximo loop
        prev_keypoints = curr_keypoints;
        prev_descriptors = curr_descriptors;
        return;
    }

    cv::Ptr<cv::flann::IndexParams> index_params = cv::makePtr<cv::flann::KDTreeIndexParams>(25); // 5 é o número de vizinhos a serem considerados

    cv::Ptr<cv::flann::SearchParams> search_params = cv::makePtr<cv::flann::SearchParams>(100); // "checks" param

    // Criar o FLANN matcher
    cv::FlannBasedMatcher matcher(index_params, search_params);
    std::vector<std::vector<cv::DMatch>> matches;

    matcher.knnMatch(prev_descriptors, curr_descriptors, matches, 2);

    // Filtrar as "boas" correspondências
    std::vector<cv::DMatch> good_matches;
    double ratio_thresh = 0.4;        // Limiar mais rigoroso para a razão de distâncias
    double max_distance = 30.0;       // Distância absoluta máxima permitida
    double max_image_distance = 10.0; // Distância máxima permitida na imagem

    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance &&
            matches[i][0].distance < max_distance)
        {
            // Calcular a distância euclidiana entre os pontos correspondentes
            cv::Point2f pt1 = prev_keypoints[matches[i][0].queryIdx].pt;
            cv::Point2f pt2 = curr_keypoints[matches[i][0].trainIdx].pt;
            double image_distance = cv::norm(pt1 - pt2);

            // Verificar se a distância na imagem está abaixo do limite
            if (image_distance < max_image_distance)
            {
                good_matches.push_back(matches[i][0]);
            }
        }
    }

    // Converter os keypoints correspondentes para pontos
    prev_points.clear();
    curr_points.clear();
    matchedPoints.clear();
    unmatchedPoints.clear();

    for (size_t i = 0; i < good_matches.size(); i++)
    {
        prev_points.push_back(prev_keypoints[good_matches[i].queryIdx].pt);
        curr_points.push_back(curr_keypoints[good_matches[i].trainIdx].pt);
        matchedPoints.push_back(curr_keypoints[good_matches[i].trainIdx].pt);
    }

    // Encontrar os keypoints sem correspondência
    for (size_t i = 0; i < curr_keypoints.size(); i++)
    {
        bool found = false;
        for (size_t j = 0; j < good_matches.size(); j++)
        {
            if (good_matches[j].trainIdx == static_cast<int>(i))
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            unmatchedPoints.push_back(curr_keypoints[i].pt);
        }
    }
    // Visualizar as correspondências filtradas
    cv::Mat img_matches;
    cv::addWeighted(prev_frame, 0.5, curr_frame, 0.5, 0, img_matches);
    // cv::drawMatches(prev_frame, prev_keypoints, curr_frame, curr_keypoints, good_matches, img_matches);
    for (size_t i = 0; i < good_matches.size(); i++)
    {
        cv::Point2f pt1 = prev_keypoints[good_matches[i].queryIdx].pt;
        cv::Point2f pt2 = curr_keypoints[good_matches[i].trainIdx].pt;
        cv::line(img_matches, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        cv::circle(img_matches, pt1, 5, cv::Scalar(255, 0, 0), -1);
        cv::circle(img_matches, pt2, 5, cv::Scalar(0, 0, 255), -1);
    }
    // Exibir a imagem com as correspondências
    cv::imshow("Good Matches", img_matches);

    calculateDistances(prev_keypoints, curr_keypoints, good_matches); // Calcula as distâncias entre as características

    // Atualizar os keypoints e descritores para o próximo loop
    prev_keypoints = curr_keypoints;
    prev_descriptors = curr_descriptors;
}

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
void MonoOdometry::drawFeatures(cv::Mat &frame, const std::vector<cv::Point2f> &matchedPoints, const std::vector<cv::Point2f> &unmatchedPoints)
{
    // Desenhar pontos correspondentes em verde
    for (const auto &point : matchedPoints)
    {
        cv::rectangle(frame, cv::Point(point.x - 5, point.y - 5), cv::Point(point.x + 5, point.y + 5), cv::Scalar(0, 255, 0), 1); // Borda verde
        cv::circle(frame, point, 1, cv::Scalar(0, 255, 0), -1);                                                                   // Círculo verde
    }
}

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
std::tuple<double, double, double> MonoOdometry::calculateDistances(const std::vector<cv::KeyPoint> &prev_keypoints, const std::vector<cv::KeyPoint> &curr_keypoints, const std::vector<cv::DMatch> &good_matches)
{
    std::vector<double> distances;
    std::vector<double> distances_x;
    std::vector<double> distances_y;

    for (size_t i = 0; i < good_matches.size(); i++)
    {
        cv::Point2f prev_pt = prev_keypoints[good_matches[i].queryIdx].pt;
        cv::Point2f curr_pt = curr_keypoints[good_matches[i].trainIdx].pt;

        // Calcular a distância euclidiana
        double distance = std::sqrt(std::pow(curr_pt.x - prev_pt.x, 2) + std::pow(curr_pt.y - prev_pt.y, 2));
        distances.push_back(distance);

        // Calcular a diferença em x e y
        double distance_x = curr_pt.x - prev_pt.x;
        double distance_y = curr_pt.y - prev_pt.y;
        distances_x.push_back(distance_x);
        distances_y.push_back(distance_y);
    }

    // Calcular a média das distâncias e dos vetores de distância em x e y
    double mean_distance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
    double mean_distance_x = std::accumulate(distances_x.begin(), distances_x.end(), 0.0) / distances_x.size();
    double mean_distance_y = std::accumulate(distances_y.begin(), distances_y.end(), 0.0) / distances_y.size();

    // Imprimir as médias calculadas
    return std::make_tuple(mean_distance, mean_distance_x, mean_distance_y);
}

/**
 * @brief Computa a odometria visual entre dois frames e atualiza a matriz de rotação e vetor de translação.
 *
 * @param prev_points Pontos do frame anterior (em escala de cinza).
 * @param curr_points Pontos do frame atual (em escala de cinza).
 * @param K Matriz de calibração da câmera.
 */
void MonoOdometry::computeOdometry(const std::vector<cv::Point2f> &prev_points, const std::vector<cv::Point2f> &curr_points, const cv::Mat &K)
{
    // Verifica se há pontos suficientes para calcular a matriz essencial
    if (prev_points.size() >= 5 && curr_points.size() >= 5)
    {
        // Estima a matriz essencial usando os pontos correspondentes e a matriz de calibração
        cv::Mat E = cv::findEssentialMat(prev_points, curr_points, K, cv::RANSAC, 0.999, 1.0);

        if (E.empty())
        {
            std::cout << "Matriz essencial vazia! Não foi possível calcular a odometria.\n";
            return;
        }

        // Matriz de rotação e vetor de translação (resultados)
        cv::Mat R_temp, t_temp;

        // Descompor a matriz essencial para obter a rotação e a translação
        cv::recoverPose(E, prev_points, curr_points, K, R_temp, t_temp);

        // Atualiza a rotação e translação acumuladas
        R = R_temp * R;       // Atualiza a rotação acumulada
        t = t + (R * t_temp); // Atualiza a translação acumulada (convertida para o sistema de coordenadas global)

        // Atualiza a posição do drone/câmera no espaço
        position += R * t_temp;

        // Exibe a rotação e translação estimadas
        std::cout << "R = " << R << std::endl;
        std::cout << "t = " << t.t() << std::endl; // Exibir a translação como uma linha
    }
    else
    {
        std::cout << "Pontos insuficientes para calcular a odometria.\n";
    }
}

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
void MonoOdometry::calculateEulerAngles(const cv::Mat &R, double &pitch, double &yaw, double &roll)
{
    // Calcular os ângulos de Euler (pitch, yaw, roll)
    pitch = atan2(R.at<double>(2, 1), R.at<double>(2, 2));                                                                     // Inclinação (pitch)
    yaw = atan2(-R.at<double>(2, 0), sqrt(R.at<double>(2, 1) * R.at<double>(2, 1) + R.at<double>(2, 2) * R.at<double>(2, 2))); // Guindagem (yaw)
    roll = atan2(R.at<double>(1, 0), R.at<double>(0, 0));                                                                      // Rolamento (roll)
}

/**
 * @brief Função principal que inicia o processamento de odometria visual.
 *
 * @param argc Número de argumentos passados na linha de comando.
 * @param argv Array de argumentos passados na linha de comando.
 * @return 0 se bem-sucedido, 1 em caso de erro.
 */
int main(int argc, char **argv)
{
    try
    {
        if (argc < 2)
        {
            std::cerr << "Usage: ./mono_odometry <video_file>" << std::endl;
            return 1;
        }

        // Obtém o nome do vídeo a partir do argumento
        std::string videoName = argv[1];

        // Obtém o caminho do vídeo
        std::filesystem::path videoPath = std::filesystem::current_path() / "video" / videoName;

        // Verifica se o arquivo de vídeo existe
        if (!std::filesystem::exists(videoPath))
        {
            std::cerr << "Erro: O arquivo de vídeo não foi encontrado: " << videoPath << std::endl;
            return 1;
        }

        std::cout << "Processando o vídeo: " << videoPath << std::endl;

        // Cria a instância de MonoOdometry e processa o vídeo
        MonoOdometry mono_odometry(videoPath.string());

        mono_odometry.processVideos();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred in main!" << std::endl;
        return 1;
    }

    return 0;
}
