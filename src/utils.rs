// utils.rs - Funzioni utili riutilizzabili per algoritmi di selezione

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::{HashMap, HashSet};

// Struttura per contenere i risultati dell'analisi MBR
#[derive(Debug)]
pub struct MbrResult {
    pub best_response: String,
    pub best_index: usize,
    pub avg_distances: Vec<f64>,
}

// Funzione helper per tokenizzare una stringa in parole (split su whitespace).
// Passo 1: Tokenizzazione semplice per bag-of-words.
pub fn tokenize(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}

// Funzione helper per calcolare la similarità coseno tra due vettori.
// Passo 2: Calcolo del prodotto scalare e delle norme per la similarità.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f64 = (a.iter().map(|&x| x * x).sum::<f64>()).sqrt();
    let norm_b: f64 = (b.iter().map(|&x| x * x).sum::<f64>()).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// Funzione helper per calcolare la distanza coseno (1 - similarità).
// Passo 3: Conversione della similarità in distanza.
pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    1.0 - cosine_similarity(a, b)
}

// Funzione per selezionare la migliore risposta basandosi sulla distanza coseno media minima.
// Assume che le risposte siano stringhe e usa un approccio bag-of-words per vettorizzare.
// Passo 4: Vettorizzazione delle risposte usando conteggi di parole (TF, senza IDF per semplicità).
// Passo 5: Calcolo delle distanze medie per ciascun candidato.
// Passo 6: Selezione dell'indice con la distanza media minima (Minimum Bayes Risk approssimato).
pub fn select_best_with_distances(outputs: &[String]) -> PyResult<MbrResult> {
    let n = outputs.len();
    if n == 0 {
        return Err(PyValueError::new_err("No outputs provided"));
    }
    if n == 1 {
        return Ok(MbrResult {
            best_response: outputs[0].clone(),
            best_index: 0,
            avg_distances: vec![0.0],
        });
    }

    // Raccogli tutte le parole uniche da tutte le risposte.
    let mut all_words: HashSet<&str> = HashSet::new();
    for out in outputs {
        for word in tokenize(out) {
            all_words.insert(word);
        }
    }

    // Converti in lista ordinata per indice consistente.
    let mut word_list: Vec<&str> = all_words.into_iter().collect();
    word_list.sort();

    // Mappa parola a indice.
    let word_to_idx: HashMap<&str, usize> = word_list.iter().enumerate().map(|(i, &w)| (w, i)).collect();

    // Crea vettori per ciascuna risposta.
    let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(n);
    for out in outputs {
        let mut vec = vec![0.0; word_list.len()];
        for word in tokenize(out) {
            if let Some(&idx) = word_to_idx.get(word) {
                vec[idx] += 1.0;
            }
        }
        vectors.push(vec);
    }

    // Trova l'indice con la distanza media minima e calcola tutte le distanze.
    let mut min_avg_dist = f64::INFINITY;
    let mut best_idx = 0;
    let mut avg_distances = Vec::with_capacity(n);
    
    for i in 0..n {
        let mut total_dist = 0.0;
        for j in 0..n {
            if i == j {
                continue;
            }
            total_dist += cosine_distance(&vectors[i], &vectors[j]);
        }
        let avg_dist = total_dist / ((n - 1) as f64);
        avg_distances.push(avg_dist);
        
        if avg_dist < min_avg_dist {
            min_avg_dist = avg_dist;
            best_idx = i;
        }
    }

    Ok(MbrResult {
        best_response: outputs[best_idx].clone(),
        best_index: best_idx,
        avg_distances,
    })
}

// Struttura per contenere i risultati del majority voting
#[derive(Debug)]
pub struct MajorityVotingResult {
    pub best_response: String,
    pub best_index: usize,
    pub vote_counts: HashMap<String, usize>,
    pub all_responses: Vec<String>,
}

// Funzione per selezionare la migliore risposta usando majority voting.
// Conta le occorrenze di ogni risposta unica e seleziona quella più frequente.
pub fn select_best_by_majority_voting(outputs: &[String]) -> PyResult<MajorityVotingResult> {
    let n = outputs.len();
    if n == 0 {
        return Err(PyValueError::new_err("No outputs provided"));
    }
    if n == 1 {
        let mut vote_counts = HashMap::new();
        vote_counts.insert(outputs[0].clone(), 1);
        return Ok(MajorityVotingResult {
            best_response: outputs[0].clone(),
            best_index: 0,
            vote_counts,
            all_responses: outputs.to_vec(),
        });
    }

    // Conta le occorrenze di ogni risposta
    let mut vote_counts: HashMap<String, usize> = HashMap::new();
    for response in outputs {
        *vote_counts.entry(response.clone()).or_insert(0) += 1;
    }

    // Trova la risposta con il maggior numero di voti
    let mut max_votes = 0;
    let mut best_response = String::new();
    for (response, count) in &vote_counts {
        if *count > max_votes {
            max_votes = *count;
            best_response = response.clone();
        }
    }

    // Trova l'indice della prima occorrenza della risposta vincente
    let best_index = outputs.iter()
        .position(|r| r == &best_response)
        .unwrap_or(0);

    Ok(MajorityVotingResult {
        best_response,
        best_index,
        vote_counts,
        all_responses: outputs.to_vec(),
    })
}

// Struttura per contenere i risultati del blender
#[derive(Debug)]
pub struct BlenderResult {
    pub best_response: String,
    pub best_index: usize,
    pub ranking_scores: Vec<f64>,
    pub fused_candidates: Vec<String>,
    pub all_responses: Vec<String>,
}

// PairRanker: Confronta coppie di risposte e assegna punteggi di ranking
// Utilizza similarità coseno per determinare quale risposta è migliore in ogni coppia
pub fn pair_ranker(outputs: &[String]) -> PyResult<Vec<f64>> {
    let n = outputs.len();
    if n == 0 {
        return Err(PyValueError::new_err("No outputs provided"));
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }

    // Raccogli tutte le parole uniche da tutte le risposte
    let mut all_words: HashSet<&str> = HashSet::new();
    for out in outputs {
        for word in tokenize(out) {
            all_words.insert(word);
        }
    }

    // Converti in lista ordinata per indice consistente
    let mut word_list: Vec<&str> = all_words.into_iter().collect();
    word_list.sort();

    // Mappa parola a indice
    let word_to_idx: HashMap<&str, usize> = word_list.iter().enumerate().map(|(i, &w)| (w, i)).collect();

    // Crea vettori per ciascuna risposta
    let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(n);
    for out in outputs {
        let mut vec = vec![0.0; word_list.len()];
        for word in tokenize(out) {
            if let Some(&idx) = word_to_idx.get(word) {
                vec[idx] += 1.0;
            }
        }
        vectors.push(vec);
    }

    // Confronto a coppie: per ogni coppia (i,j), determina quale ha punteggio migliore
    let mut win_counts = vec![0.0; n];
    
    for i in 0..n {
        for j in (i+1)..n {
            let _similarity = cosine_similarity(&vectors[i], &vectors[j]);
            
            // Calcola la qualità relativa basata sulla lunghezza e diversità
            let len_i = outputs[i].len() as f64;
            let len_j = outputs[j].len() as f64;
            let diversity_i = vectors[i].iter().filter(|&&x| x > 0.0).count() as f64;
            let diversity_j = vectors[j].iter().filter(|&&x| x > 0.0).count() as f64;
            
            // Punteggio combinato: lunghezza + diversità lessicale
            let score_i = len_i * 0.3 + diversity_i * 0.7;
            let score_j = len_j * 0.3 + diversity_j * 0.7;
            
            if score_i > score_j {
                win_counts[i] += 1.0;
            } else if score_j > score_i {
                win_counts[j] += 1.0;
            } else {
                // Pareggio: entrambi ottengono 0.5 punti
                win_counts[i] += 0.5;
                win_counts[j] += 0.5;
            }
        }
    }

    // Normalizza i punteggi
    let total_comparisons = (n * (n - 1)) as f64 / 2.0;
    let ranking_scores: Vec<f64> = win_counts.iter()
        .map(|&count| count / total_comparisons)
        .collect();

    Ok(ranking_scores)
}

// GenFuser: Fonde le migliori candidate basandosi sui punteggi di ranking
// Seleziona le top-k risposte e le combina in modo intelligente
pub fn gen_fuser(outputs: &[String], ranking_scores: &[f64], top_k: usize) -> PyResult<Vec<String>> {
    let n = outputs.len();
    if n == 0 {
        return Err(PyValueError::new_err("No outputs provided"));
    }
    
    let k = std::cmp::min(top_k, n);
    
    // Crea coppie (indice, punteggio) e ordina per punteggio decrescente
    let mut indexed_scores: Vec<(usize, f64)> = ranking_scores.iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();
    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Prendi le top-k risposte
    let top_indices: Vec<usize> = indexed_scores.iter()
        .take(k)
        .map(|(idx, _)| *idx)
        .collect();
    
    let mut fused_candidates = Vec::new();
    
    // Strategia 1: Risposta con punteggio più alto
    if let Some(&best_idx) = top_indices.first() {
        fused_candidates.push(outputs[best_idx].clone());
    }
    
    // Strategia 2: Concatenazione delle frasi più importanti dalle top-k
    if k > 1 {
        let mut combined_sentences = Vec::new();
        
        for &idx in &top_indices {
            let sentences: Vec<&str> = outputs[idx].split('.').collect();
            for sentence in sentences {
                let trimmed = sentence.trim();
                if !trimmed.is_empty() && trimmed.len() > 10 {
                    combined_sentences.push(trimmed.to_string());
                }
            }
        }
        
        // Rimuovi duplicati e ordina per lunghezza
        combined_sentences.sort();
        combined_sentences.dedup();
        combined_sentences.sort_by(|a, b| b.len().cmp(&a.len()));
        
        // Prendi le prime 3 frasi più lunghe
        let best_sentences: Vec<String> = combined_sentences.iter()
            .take(3)
            .cloned()
            .collect();
        
        if !best_sentences.is_empty() {
            fused_candidates.push(best_sentences.join(". ") + ".");
        }
    }
    
    // Strategia 3: Media pesata delle parole chiave
    if k > 2 {
        let mut word_weights: HashMap<String, f64> = HashMap::new();
        
        for &idx in &top_indices {
            let weight = ranking_scores[idx];
            for word in tokenize(&outputs[idx]) {
                *word_weights.entry(word.to_string()).or_insert(0.0) += weight;
            }
        }
        
        // Seleziona le parole con peso più alto
        let mut weighted_words: Vec<(String, f64)> = word_weights.into_iter().collect();
        weighted_words.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let top_words: Vec<String> = weighted_words.iter()
            .take(20)
            .map(|(word, _)| word.clone())
            .collect();
        
        if !top_words.is_empty() {
            fused_candidates.push(format!("Key concepts: {}", top_words.join(", ")));
        }
    }
    
    Ok(fused_candidates)
}

// Funzione principale per il blender che combina PairRanker e GenFuser
pub fn select_best_with_blender(outputs: &[String], top_k: Option<usize>) -> PyResult<BlenderResult> {
    let n = outputs.len();
    if n == 0 {
        return Err(PyValueError::new_err("No outputs provided"));
    }
    if n == 1 {
        return Ok(BlenderResult {
            best_response: outputs[0].clone(),
            best_index: 0,
            ranking_scores: vec![1.0],
            fused_candidates: vec![outputs[0].clone()],
            all_responses: outputs.to_vec(),
        });
    }

    let k = top_k.unwrap_or(std::cmp::min(3, n));
    
    // Fase 1: PairRanker - calcola i punteggi di ranking
    let ranking_scores = pair_ranker(outputs)?;
    
    // Fase 2: GenFuser - genera candidate fuse
    let fused_candidates = gen_fuser(outputs, &ranking_scores, k)?;
    
    // Seleziona la migliore risposta basandosi sul punteggio più alto
    let best_index = ranking_scores.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    
    // Se abbiamo candidate fuse, usa la prima come migliore risposta
    let best_response = if !fused_candidates.is_empty() {
        fused_candidates[0].clone()
    } else {
        outputs[best_index].clone()
    };

    Ok(BlenderResult {
        best_response,
        best_index,
        ranking_scores,
        fused_candidates,
        all_responses: outputs.to_vec(),
    })
}
