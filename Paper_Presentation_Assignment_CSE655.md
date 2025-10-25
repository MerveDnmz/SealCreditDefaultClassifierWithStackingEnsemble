# CSE 655 – Paper Presentation Assignment

**Student:** Merve DÖNMEZ (244201001016)  
**Course:** Deep Learning and Applications (CSE 655)  
**Presentation Duration:** 15-20 minutes

---

## Selected Papers for Presentation

### Paper 1: "Practical considerations of fully homomorphic encryption in privacy-preserving machine learning"
- **Authors:** D. C.-T. Lo, Y. Shi, H. Shahriar, B. Deng, X. Zhang, M.-L. Chen
- **Venue:** IEEE BigData Conference 2024
- **DOI:** 10.1109/BigData62323.2024.10825068
- **Relevance:** Directly addresses FHE implementation challenges in ML, matches our CKKS/SEAL integration work

### Paper 2: "Performance comparison of homomorphic encrypted convolutional neural network inference between Microsoft SEAL and OpenFHE"
- **Authors:** H. Zhu, T. Suzuki, H. Huang, H. Yamana
- **Venue:** DEIM 2023 (Forum on Data Engineering and Information Management)
- **Link:** https://proceedingsofdeim.github.io/DEIM2023/5b-9-2.pdf
- **Relevance:** Compares HE libraries performance, supports our SEAL-based implementation

### Paper 3: "HEProfiler: An in-depth profiler of approximate homomorphic encryption libraries"
- **Authors:** J. Takeshita, N. Koirala, C. McKechney, T. Jung
- **Venue:** Research Square (Preprint) 2022
- **DOI:** 10.21203/rs.3.rs-2164106/v1
- **Relevance:** Provides benchmarking framework for CKKS-based libraries, validates our performance analysis approach

---

## Presentation Outline (15-20 minutes)

### 1. Introduction & Motivation (2-3 minutes)
**Problem Statement:**
- Financial data privacy concerns in machine learning
- Need for privacy-preserving credit default prediction
- Regulatory compliance requirements (GDPR, financial regulations)

**Motivation:**
- Our project: CKKS-based encrypted inference for credit risk modeling
- Deep learning models (Dense, Transformer, Hybrid) vs traditional ML
- Performance trade-offs: accuracy vs privacy vs computational cost

### 2. Paper 1: Practical FHE Considerations (4-5 minutes)
**Problem & Motivation:**
- FHE implementation challenges in real-world ML applications
- Parameter selection complexity (scale, modulus, polynomial degree)
- Computational overhead vs privacy benefits

**Dataset & Methodology:**
- Various ML tasks with different data types
- CKKS parameter optimization strategies
- Performance benchmarking across different scenarios

**Key Results:**
- Accuracy preservation under encryption
- Significant computational overhead (10-100x slower)
- Memory consumption increases
- Batch processing optimization recommendations

**Strengths & Limitations:**
- ✅ Comprehensive practical guidance
- ✅ Real-world implementation insights
- ❌ Limited to specific HE schemes
- ❌ Hardware-specific optimizations

**Connection to Our Work:**
- Directly validates our CKKS parameter choices (poly_modulus_degree=4096, scale=2^35)
- Supports our batch_size=256 optimization
- Confirms accuracy preservation in our results (Dense: 0.7206→0.7206, Transformer: 0.7269→0.7269)

### 3. Paper 2: SEAL vs OpenFHE Performance (4-5 minutes)
**Problem & Motivation:**
- Need for systematic comparison of HE libraries
- CNN inference performance under different HE implementations
- Library selection criteria for ML applications

**Dataset & Methodology:**
- MNIST, CIFAR-10 datasets
- CNN models with different architectures
- SEAL vs OpenFHE performance metrics (latency, throughput, memory)

**Key Results:**
- SEAL superior performance in low-depth computations
- OpenFHE better for complex operations
- Library-specific optimization strategies
- Hardware acceleration benefits

**Strengths & Limitations:**
- ✅ Direct library comparison
- ✅ Hardware acceleration analysis
- ❌ Limited to image data
- ❌ Single model architecture focus

**Connection to Our Work:**
- Validates our SEAL library choice for tabular data
- Supports our performance optimization approach
- Our results show SEAL works well for credit default prediction (AUC: 0.8010-0.8107)

### 4. Paper 3: HEProfiler Benchmarking (3-4 minutes)
**Problem & Motivation:**
- Need for systematic HE library profiling
- Performance bottleneck identification
- Optimization strategy development

**Dataset & Methodology:**
- Multiple CKKS-based libraries (SEAL, PALISADE, HElib, HEAAN)
- Primitive operation benchmarking
- Bootstrapping cost analysis
- Multi-threading capabilities

**Key Results:**
- Detailed performance profiling framework
- Library-specific optimization recommendations
- Multi-threading performance gains
- Bootstrapping cost analysis

**Strengths & Limitations:**
- ✅ Comprehensive profiling framework
- ✅ Multi-library comparison
- ❌ Limited to primitive operations
- ❌ No end-to-end ML pipeline analysis

**Connection to Our Work:**
- Supports our batch processing optimization (batch_size=256)
- Validates our performance measurement approach
- Our implementation shows similar optimization patterns

### 5. Critical Analysis & Our Contributions (2-3 minutes)
**Critical Analysis:**
- **Paper 1:** Excellent practical guidance but limited to specific scenarios
- **Paper 2:** Good library comparison but focused on image data
- **Paper 3:** Comprehensive profiling but lacks ML pipeline analysis

**Our Project Contributions:**
- **Novel Application:** First comprehensive deep learning + CKKS implementation for credit default prediction
- **Model Diversity:** Dense, Transformer, Hybrid models vs traditional Stacking
- **Performance Analysis:** Detailed comparison of encrypted vs unencrypted inference
- **Practical Insights:** Batch optimization, parameter tuning, error handling

**Key Findings from Our Work:**
- **Accuracy Preservation:** All models maintain accuracy under encryption
- **Performance Ranking:** Stacking (0.8795 AUC) > Transformer (0.8107) > Hybrid (0.8048) > Dense (0.8010)
- **Encryption Overhead:** Minimal impact on model performance
- **Optimization Success:** Batch_size=256 provides optimal performance

### 6. Future Work & Conclusion (1-2 minutes)
**Future Directions:**
- Full encrypted training (not just inference)
- GPU acceleration for HE operations
- Federated learning integration
- Real-time deployment considerations

**Conclusion:**
- Deep learning models show competitive performance in privacy-preserving credit risk modeling
- CKKS encryption preserves model accuracy with manageable computational overhead
- Our work bridges the gap between theoretical HE research and practical ML applications

---

## References

1. Lo, D. C.-T., Shi, Y., Shahriar, H., Deng, B., Zhang, X., & Chen, M.-L. (2024). Practical considerations of fully homomorphic encryption in privacy-preserving machine learning. *Proceedings of the 2024 IEEE International Conference on Big Data*, 5181-5190. https://doi.org/10.1109/BigData62323.2024.10825068

2. Zhu, H., Suzuki, T., Huang, H., & Yamana, H. (2023). Performance comparison of homomorphic encrypted convolutional neural network inference between Microsoft SEAL and OpenFHE. *Proceedings of the 15th Forum on Data Engineering and Information Management*, Tokyo, Japan. https://proceedingsofdeim.github.io/DEIM2023/5b-9-2.pdf

3. Takeshita, J., Koirala, N., McKechney, C., & Jung, T. (2022). HEProfiler: An in-depth profiler of approximate homomorphic encryption libraries. *Research Square*. https://doi.org/10.21203/rs.3.rs-2164106/v1

---

## Presentation Notes

### Key Talking Points:
1. **Start with our project results** - show the performance comparison table
2. **Connect each paper to our implementation** - specific parameter choices, optimizations
3. **Highlight our unique contributions** - deep learning + CKKS for tabular data
4. **Use visual aids** - ROC curves, training history, performance charts
5. **Emphasize practical impact** - real-world financial data, regulatory compliance

### Visual Aids to Include:
- Performance comparison table (our results)
- ROC curves comparison
- Training history plots
- Architecture diagrams (Dense, Transformer, Hybrid)
- CKKS encryption/decryption flow diagram

### Critical Analysis Points:
- **Gap in Literature:** Most HE+ML work focuses on image data, we address tabular financial data
- **Practical Implementation:** We provide working code with error handling and optimization
- **Performance Validation:** Our results confirm theoretical predictions from the papers
- **Real-world Application:** Credit default prediction has immediate practical value
