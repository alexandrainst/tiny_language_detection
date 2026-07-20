# Literature Survey: Automatic Language Detection of Audio on Edge Devices

## Introduction

Automatic language detection (ALD) of audio is a foundational technology enabling a wide range of multilingual applications, from real-time translation to voice assistants and accessibility systems. With the proliferation of edge devices—such as smartphones, wearables, microcontrollers, and embedded systems—there is a growing demand for ALD solutions that can operate efficiently and accurately without relying on cloud infrastructure. Running ALD on edge devices enhances privacy, reduces latency, and ensures reliable operation even in bandwidth-constrained or offline environments. However, deploying ALD on resource-constrained hardware presents significant challenges, including limited computational power, memory, and energy availability. This survey synthesizes recent advancements in models, feature extraction, optimization techniques, and system architectures for automatic language detection of audio running on edge devices.

## Feature Extraction and Model Architectures for Edge Deployment

Efficient feature extraction is critical for enabling real-time ALD on edge hardware. Mel-Frequency Cepstral Coefficients (MFCCs), Log Mel-Spectrograms, Short-Time Fourier Transform (STFT), and Discrete Wavelet Transform (DWT) are widely used due to their balance of computational efficiency and representational power (Liu, 2026; Darsi et al., 2026; Fahim et al., 2025). MFCCs, in particular, offer fast processing and low memory usage, making them ideal for embedded systems and microcontrollers (Darvishi, 2026; Patil et al., 2024). DWT and Continuous Wavelet Transform (CWT) provide higher accuracy but at greater computational cost, which can be mitigated through hardware acceleration or model optimization (Fahim et al., 2025; Bin Liu et al., 2026).

Model architectures for ALD on edge devices have evolved to maximize accuracy while minimizing resource consumption. Lightweight convolutional neural networks (CNNs), recurrent neural networks (RNNs), and hybrid CNN-RNN models are commonly adopted (Cerna et al., 2023; Ezilarasan et al., 2026). Spiking neural networks (SCNNs) and analog-spiking hybrids further reduce energy usage, leveraging event-based processing and neuromorphic hardware for real-time inference (Leow et al., 2023; Ponghiran and Roy, 2021; Bin Liu et al., 2026). Model order reduction, pruning, quantization, and knowledge distillation techniques are essential for compressing models to fit within the stringent memory and computation budgets of edge platforms (Mou and Milanova, 2024; Bittner et al., 2025; Md Mohaimenuzzaman et al., 2021; Liu, 2026).

Recent works also explore transformer-based models and large audio language models (LALMs), using token compression, margin-based contrastive learning, and layer-wise optimization to retain performance while reducing complexity (Rezaabad et al., 2025; Bhati et al., 2025). Specialized architectures, such as M3Net with mirror attention and PolyLingua, demonstrate that high accuracy (over 97–98% F1) is achievable with a fraction of the parameters of traditional models (Jiang et al., 2025; Rezaabad et al., 2025).

## System Optimization and Real-Time Edge Inference

Achieving real-time ALD on edge devices requires not only efficient models but also system-level optimizations. Dynamic quantization, model switching, and adaptive scheduling can significantly reduce inference times and resource usage, as demonstrated by Mekonnen et al. (2025) and Shah et al. (2025). Systems like Dyn-ASR dynamically select and load compact, accent- or language-specific ASR models, minimizing memory footprint and maximizing accuracy for multilingual and multi-accent scenarios (Ghangam et al., 2021). Streaming architectures, such as WhisperPipe, employ hybrid voice activity detection and dynamic buffering to maintain low latency and stable resource usage during continuous audio processing (Ramezani et al., 2026).

Hardware acceleration further enables complex models to run efficiently on edge devices. FPGA-based deployments of CNN-LSTM and graph neural networks achieve fast inference and low power consumption, making them suitable for real-time ALD in embedded environments (Ezilarasan et al., 2026; Jeziorek et al., 2026). TinyML platforms and frameworks like Edge Impulse and acoupi facilitate the deployment, management, and real-time execution of ALD models on microcontrollers and single-board computers (Patil et al., 2024; Vuilliomenet et al., 2025).

Privacy and accessibility are also enhanced by running ALD locally. Applications include voice user interfaces for people with speech impairments (Mulfari and Villari, 2024), abusive language detection (Zhu et al., 2025; Ramadan et al., 2022), and human-robot interaction (Albuali et al., 2026; Łukawski et al., 2025). These systems demonstrate robust performance under real-world conditions, including noise, varied accents, and speech disorders, and are tailored for deployment in environments where privacy and reliability are paramount.

## Multilingual, Multimodal, and Specialized Applications

Edge-based ALD systems increasingly support multilingual, code-switching, and multimodal scenarios. Models like FunAudioLLM and SenseVoice enable multilingual speech recognition and language detection, with small model variants optimized for edge deployment (An et al., 2024). Real-time translation frameworks combine ASR, neural machine translation, and TTS, leveraging compressed transformer models and mixed-precision quantization for wearable devices and robotics (Khan, 2025; Albuali et al., 2026; Vikas et al., 2025).

Multimodal integration—combining audio, vision, and text—enhances context-aware language detection and user interaction on edge devices (Nath et al., 2025; Mekonnen et al., 2025). Systems like Multi-Modal Fusion and MHR-RAG support audio-visual reasoning and align audio inputs with large language models, achieving low-latency, resource-efficient multilingual understanding (Nath et al., 2025; Bilvitha et al., 2026; Qin et al., 2024).

Specialized applications address low-resource and underrepresented languages, code-switching, and overlapping speech. Synthetic data augmentation using multilingual TTS improves LID performance in low-resource settings (Dey et al., 2026), while modular pipelines separate overlapping speakers before language recognition to handle complex multilingual audio (Kolsur et al., 2025; Jani et al., 2024). Few-shot learning and adaptive embedding techniques enable rapid adaptation to new languages and keywords, supporting scalable and cost-effective deployment (L. B, 2025).

## Conclusion

Automatic language detection of audio on edge devices has seen significant advancements, driven by innovations in feature extraction, model compression, and system optimization. Lightweight architectures, efficient feature representations, and hardware acceleration enable high-accuracy, low-latency language detection even on microcontrollers and embedded platforms. System-level techniques such as dynamic model switching, streaming inference, and TinyML frameworks further bridge the gap between performance and resource constraints. Multilingual, multimodal, and specialized solutions are extending the reach of edge-based ALD to diverse real-world applications, including accessibility, safety moderation, and human-robot interaction. Future research should continue to focus on adaptive, robust, and scalable models, especially for low-resource languages and challenging acoustic environments, ensuring that automatic language detection remains accessible, privacy-preserving, and effective across the expanding landscape of edge devices.

## References

Albuali, Abdullah and Tripathi, Nishant and R., M. and Almusharraf, Ahlam and Abrar, Muhammad and Siddiqui, Isma (2026). An Edge-Enabled Low-Latency Cross-Lingual Speech-To-Text Framework For Efficient Human-Robot Interaction.. _Big Data_.

An, Keyu and Chen, Qian and Deng, Chong and Du, Zhihao and Gao, Changfeng and Gao, Zhifu and Gu, Yue and He, Ting and Hu, Hangrui and Hu, Kai and Ji, Shengpeng and Li, Yabin and Li, Zerui and Lu, Heng and Lv, Xiang and Ma, Bin and Ma, Ziyang and Ni, Chongjia and Song, Changhe and Shi, Jiaqi and Shi, Xian and Wang, Hao and Wang, Wen and Wang, Yuxuan and Xiao, Zhangyu and Yan, Zhijie and Yang, Yexin and Zhang, Bin and Zhang, Qingling and Zhang, Shiliang and Zhao, Nan and Zheng, Siqi (2024). Funaudiollm: Voice Understanding And Generation Foundation Models For Natural Interaction Between Humans And Llms. _Arxiv.Org_.

B, L. (2025). Keyword Spotting System. _International Journal For Research In Applied Science And Engineering Technology_.

Bhati, Saurabhchand and Thomas, Samuel and Kuehne, Hildegard and Feris, Rogério and Glass, James (2025). Towards Audio Token Compression In Large Audio Language Models. _Arxiv.Org_.

Bilvitha, Devarapalli and Isravel, Deva and Dhas, J. (2026). Resource-Efficient Multilingual Rag: English-Only And Multilingual Embeddings For Indian Languages.

Bittner, Matthias and Schnöll, Daniel and Dallinger, Dominik and Wess, M. and Jantsch, Axel (2025). Pruning State Space Models With Model Order Reduction For Efficient Raw Audio Classification. _European Signal Processing Conference_.

Cerna, P. and Ututalum, Charisma and Evangelista, R. and Darkis, Aldaruhz and Asiri, M. and Muallam-Darkis, Jehana (2023). An Iot-Based Language Recognition System For Indigenous Languages Using Integrated Cnn And Rnn.

Darsi, Loukik and Shubham, Shubham and Pothuri, Avaneesh and Nagarajan, Ponnalagu and Gupta, Manik (2026). Edge Intelligence For Speech: Evaluating Model Performance, Feature Extraction And Signal Fidelity On Commodity Hardware. _International Conference On Communication Systems And Networks_.

Darvishi, Mostafa (2026). Embedded Machine Learning For Microcontroller-Class Edge Devices: Data, Feature, Evaluation, And Deployment Pipelines.

Dey, Spandan and Prasad, Karthik and Kumari, Surbhi and Dash, Debi and Acharya, U. and Muduli, Himanshu and Sahu, Ashutosh and Agrawal, Gopal and Maheshwari, Bharat and Routray, Rashmita (2026). Multilingual Tts For Improving Spoken Language Identification With Low-Resource Assumptions. _National Conference On Communications_.

Ezilarasan, M. and Kavitha, G. and G, Rahul and Che, Hangjun and Dai, Xiangguang and Feng, Yuming and Leung, Man-Fai (2026). Fpga-Based Cnn-Lstm System For Audio Signal Classification. _International Conference On Intelligent Control And Information Processing_.

Fahim, Haider and Kalidoss, D. and Kizi, K. and Jaber, Abdullah (2025). Wavelet Transform Enhanced Signal Processing For Low-Latency Audio Recognition In Edge Devices. _International Conference Control And Robots_.

Ghangam, Sangeeta and Whitenack, Daniel and Nemecek, Joshua (2021). Dyn-Asr: Compact, Multilingual Speech Recognition Via Spoken Language And Accent Identification. _World Forum On Internet Of Things_.

Jani, Mayur and Panchal, Sandip and Patel, Hemant and Sureja, Nitesh and Raiyani, A. (2024). Automated Detection And Translation Of Multilingual Speech: A System For Real-Time Language Recognition And Conversion. _Journal Of Electrical Systems_.

Jeziorek, K. and Wzorek, Piotr and Błachut, Krzysztof and Nakano, Hiroshi and Dampfhoffer, M. and Mesquida, Thomas and Nishi, Hiroaki and Dalgaty, Thomas and Kryjak, Tomasz (2026). Hardware-Accelerated Graph Neural Networks: An Alternative Approach For Neuromorphic Event-Based Audio Classification And Keyword Spotting On Soc Fpga. _Arxiv.Org_.

Jiang, Xuanming and An, Baoyi and Zhao, Guoshuai and Qian, Xueming (2025). M3Net: Efficient Time-Frequency Integration Network With Mirror Attention For Audio Classification On Edge. _Aaai Conference On Artificial Intelligence_.

Khan, M. (2025). Real-Time Speech Translation For Wearable Devices: A Multi-Modal Approach Using Edge Computing And Neural Machine Translation. _International Journal For Research In Applied Science And Engineering Technology_.

Kolsur, Anupama and Prajwal, K. and Vijayasenan, Deepu (2025). Language Detection In Overlapping Multilingual Speech: A Focus On Indian Languages.

Leow, Cong and Goh, W. and Gao, Yuan (2023). Sparsity Through Spiking Convolutional Neural Network For Audio Classification At The Edge. _International Symposium On Circuits And Systems_.

Liu, Donghao (2026). A Survey On Lightweight Audio Classification For Edge Devices. _Applied And Computational Engineering_.

Liu, Bin and Li, Wenjuan and Li, Bing and Yuan, C. and Shang, Kun and Gao, Shaobing and Hu, Weiming (2026). Wavespikenet: A Wavelet-Spiking Fusion Architecture For Audio Classification On Edge Devices. _Ieee International Conference On Acoustics, Speech, And Signal Processing_.

Mekonnen, Fitsum and Bataineh, M. and Abdoun, Dana and Serag, Ahmed and Tamiru, Kena and Abula, Winner and Darota, Simon (2025). Offline Multimodal Edge Ai Framework Integrating Computer Vision And Speech Processing For Ict Accessibility Systems. _Advanced Industrial Conference On Telecommunications_.

Mohaimenuzzaman, Md and Bergmeir, C. and Meyer, B. (2021). Pruning Vs Xnor-Net: A Comprehensive Study Of Deep Learning For Audio Classification On Edge-Devices. _Ieee Access_.

Mou, Afsana and Milanova, M. (2024). Performance Analysis Of Deep Learning Model-Compression Techniques For Audio Classification On Edge Devices. _The Scientist_.

Mulfari, Davide and Villari, Massimo (2024). A Voice User Interface On The Edge For People With Speech Impairments. _Electronics_.

Nath, Kshitish and Kumar, Priyaranjan and Leslie, Reshma and Jain, Pankajkumar (2025). Multi-Modal Fusion On The Edge: Real-Time Audio-Visual Llms For Smart Devices.

Patil, Malhar and Rawoorkar, Prajwal and Muley, P. and Motade, Sumitra and Kukade, Shweta and Deshpande, Anagha and Nair, Arunkumar (2024). Edge Impulse: Tinyml Language Classification Model.

Ponghiran, Wachirawit and Roy, K. (2021). Hybrid Analog-Spiking Long Short-Term Memory For Energy Efficient Computing On Edge Devices. _Design, Automation And Test In Europe_.

Qin, Ruiyang and Liu, Dancheng and Xu, Gelei and Yan, Zheyu and Xu, Chenhui and Hu, Yuting and Hu, X. and Xiong, Jinjun and Shi, Yiyu (2024). Tiny-Align: Bridging Automatic Speech Recognition And Large Language Model On Edge.

Ramadan, Syed and Sakib, T. and Rahat, Md. and Hossain, Md. and Rahman, Raiyan and Rahman, Md. (2022). An Integrated Embedded System Towards Abusive Bengali Speech And Speaker Detection Using Nlp And Deep Learning.

Ramezani, E. and Giahi, Mohammad and Zarabadipour, Mohammad and Yosefian, Amir and Ghadiri, Hamidreza (2026). Whisperpipe: A Resource-Efficient Streaming Architecture For Real-Time Automatic Speech Recognition. _Arxiv.Org_.

Rezaabad, Ali and Khanal, Bikram and Chaurasia, S. and Zeng, Lu and Hong, Dezhi and Bashashati, Hossein and Butler, Thomas and Ganji, Megan (2025). Polylingua: Margin-Based Inter-Class Transformer For Robust Cross-Domain Language Detection. _Arxiv.Org_.

Shah, Aatman and Konidena, Dedipya and Srivastava, Satyam and Nehra, Shalini and Mukhiya, Suraj (2025). A Novel Dynamic Model Switching Approach For Deep Learning On Edge Devices With Performance Evaluation. _International Symposium On Embedded Computing And System Design_.

Vuilliomenet, Aude and Balvanera, Santiago and Aodha, Oisin and Jones, Kate and Wilson, D. (2025). Acoupi: An Open‐Source Python Framework For Deploying Bioacoustic Ai Models On Edge Devices. _Methods In Ecology And Evolution_.

Zhu, Yi-Chang and Hung, Ying-Hsiu and Chang, Yen-Ching and Tang, Jhen-Kai and Tsai, Wen-Kai and Lai, Shin-Chi (2025). Speech Abusive Language Detection System Using Mfcc Speech Feature Extraction And Convolutional Neural Network.

Łukawski, Bartek and Victores, J. and Balaguer, Carlos and Jardón, A. (2025). Interaction With A Humanoid Robot Through A Conversational Interface Using Deepseek. _Workshop On Multimedia For Cooking And Eating Activities_.
