// lib.rs

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyString, PyCFunction, PyList, PyFloat};
use std::collections::{HashMap, HashSet};
use pyo3::exceptions::PyValueError;
use std::time::{Instant, Duration};

// Funzione helper per tokenizzare una stringa in parole (split su whitespace).
// Passo 1: Tokenizzazione semplice per bag-of-words.
fn tokenize(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}

// Funzione helper per calcolare la similarità coseno tra due vettori.
// Passo 2: Calcolo del prodotto scalare e delle norme per la similarità.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
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
fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    1.0 - cosine_similarity(a, b)
}

// Struttura per contenere i risultati dell'analisi MBR
#[derive(Debug)]
struct MbrResult {
    best_response: String,
    best_index: usize,
    avg_distances: Vec<f64>,
}

// Funzione per selezionare la migliore risposta basandosi sulla distanza coseno media minima.
// Assume che le risposte siano stringhe e usa un approccio bag-of-words per vettorizzare.
// Passo 4: Vettorizzazione delle risposte usando conteggi di parole (TF, senza IDF per semplicità).
// Passo 5: Calcolo delle distanze medie per ciascun candidato.
// Passo 6: Selezione dell'indice con la distanza media minima (Minimum Bayes Risk approssimato).
fn select_best_with_distances(outputs: &[String]) -> PyResult<MbrResult> {
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

// Funzione helper per creare il wrapper della funzione decorata.
// Passo 7: Uso di multiprocessing per eseguire chiamate parallele (vero parallelismo, bypassando GIL).
// Passo 8: Raccolta degli output come stringhe.
// Passo 9: Selezione del migliore usando select_best e restituzione.
fn create_wrapper(py: Python, func: PyObject, num_calls: usize) -> PyResult<PyObject> {
    let closure = PyCFunction::new_closure(py, None, None, move |args, kwargs| {
        Python::with_gil(|py| {
            let start_time = Instant::now();
            
            // Esegui le chiamate sequenzialmente per evitare problemi di pickling.
            let mut outputs: Vec<String> = Vec::with_capacity(num_calls);
            let mut individual_times: Vec<f64> = Vec::with_capacity(num_calls);

            for _ in 0..num_calls {
                let call_start = Instant::now();
                
                // Chiama la funzione con gli argomenti originali.
                let result = func.call(py, args, kwargs)?;
                // Assume che la funzione restituisca una stringa (risposta LLM).
                let response = result.extract::<String>(py)?;
                
                let call_duration = call_start.elapsed();
                outputs.push(response);
                individual_times.push(call_duration.as_secs_f64());
            }

            // Seleziona il migliore con le distanze.
            let mbr_result = select_best_with_distances(&outputs)?;
            let total_time = start_time.elapsed();

            // Crea il dizionario di output.
            let result_dict = PyDict::new(py);
            
            // Risposta selezionata
            result_dict.set_item("selected_response", PyString::new(py, &mbr_result.best_response))?;
            
            // Tempo totale di esecuzione
            result_dict.set_item("total_execution_time", PyFloat::new(py, total_time.as_secs_f64()))?;
            
            // Tutte le risposte con i loro tempi
            let responses_list = PyList::empty(py);
            for (i, (response, time)) in outputs.iter().zip(individual_times.iter()).enumerate() {
                let response_dict = PyDict::new(py);
                response_dict.set_item("index", i)?;
                response_dict.set_item("response", PyString::new(py, response))?;
                response_dict.set_item("execution_time", PyFloat::new(py, *time))?;
                response_dict.set_item("avg_distance", PyFloat::new(py, mbr_result.avg_distances[i]))?;
                responses_list.append(response_dict)?;
            }
            result_dict.set_item("all_responses", responses_list)?;
            
            // Indice della risposta selezionata
            result_dict.set_item("selected_index", mbr_result.best_index)?;
            
            // Distanze medie per MBR
            let distances_list = PyList::empty(py);
            for distance in &mbr_result.avg_distances {
                distances_list.append(PyFloat::new(py, *distance))?;
            }
            result_dict.set_item("mbr_distances", distances_list)?;

            // Restituisci il dizionario.
            Ok::<PyObject, PyErr>(result_dict.into())
        })
    })?;
    Ok(closure.into())
}

// La funzione principale esposta come decoratore.
// Gestisce sia @minimum_bayes_risk (default num_calls=5) che @minimum_bayes_risk(num_calls=N).
// Passo 10: Logica per distinguere chiamata con argomenti (restituisce decoratore) o diretta (decora con default).
#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn minimum_bayes_risk(py: Python, args: &Bound<PyTuple>, kwargs: Option<&Bound<PyDict>>) -> PyResult<PyObject> {
    let default_num_calls = 5;

    // Se chiamata senza args o con func come primo arg (caso @minimum_bayes_risk).
    if args.len() == 1 && args.get_item(0)?.is_callable() {
        let func = args.get_item(0)?.clone().into();
        return create_wrapper(py, func, default_num_calls);
    }

    // Altrimenti, parse num_calls da kwargs o args, e restituisci un decoratore.
    let mut num_calls = default_num_calls;
    if let Some(kw) = kwargs {
        if let Some(n) = kw.get_item("num_calls")? {
            num_calls = n.extract::<usize>()?;
        }
    } else if args.len() == 1 {
        num_calls = args.get_item(0)?.extract::<usize>()?;
    }

    // Restituisci una closure che agisce come decoratore.
    let decorator = PyCFunction::new_closure(py, None, None, move |dec_args, _dec_kwargs| {
        Python::with_gil(|py| {
            if dec_args.len() != 1 || !dec_args.get_item(0)?.is_callable() {
                return Err(PyValueError::new_err("Decorator expects a callable"));
            }
            let func = dec_args.get_item(0)?.clone().into();
            create_wrapper(py, func, num_calls)
        })
    })?;
    Ok(decorator.into())
}

// Definizione del modulo Python.
// Passo 11: Registrazione della funzione decoratore nel modulo.
#[pymodule]
fn transforest(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(minimum_bayes_risk, m)?)?;
    Ok(())
}