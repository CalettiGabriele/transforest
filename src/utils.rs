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
