// lib.rs - Decoratore Minimum Bayes Risk per Python

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyString, PyCFunction, PyList, PyFloat};
use pyo3::exceptions::PyValueError;
use std::time::Instant;

mod utils;
use crate::utils::{select_best_with_distances, select_best_by_majority_voting, select_best_with_ai_blender, InferenceConfig};

// Funzione helper per creare il wrapper della funzione decorata.
// Passo 7: Uso di multiprocessing per eseguire chiamate parallele (vero parallelismo, bypassando GIL).
// Passo 8: Raccolta degli output come stringhe.
// Passo 9: Selezione del migliore usando select_best e restituzione.
fn create_wrapper(py: Python, func: PyObject, num_calls: usize) -> PyResult<PyObject> {
    let closure = PyCFunction::new_closure(py, None, None, move |args, kwargs| {
        Python::with_gil(|py| {
            let start_time = Instant::now();

            // Prepara oggetti trasferibili tra thread
            let args_owned = args.clone().unbind();
            let kwargs_owned = kwargs.map(|kw| kw.clone().unbind());
            let func_obj = func.clone_ref(py);

            // Prepara specifiche di chiamata per ogni thread
            let mut call_specs: Vec<(PyObject, Py<PyTuple>, Option<Py<PyDict>>)> = Vec::with_capacity(num_calls);
            for _ in 0..num_calls {
                let f_clone = func_obj.clone_ref(py);
                let a_clone = args_owned.clone_ref(py);
                let k_clone = kwargs_owned.as_ref().map(|kk| kk.clone_ref(py));
                call_specs.push((f_clone, a_clone, k_clone));
            }

            // Esegui le chiamate in parallelo, ciascun thread acquisisce il GIL solo per la call
            let (outputs, individual_times): (Vec<String>, Vec<f64>) = py.allow_threads(|| -> PyResult<(Vec<String>, Vec<f64>)> {
                let mut handles = Vec::with_capacity(num_calls);
                for (f, a, k) in call_specs.into_iter() {
                    handles.push(std::thread::spawn(move || {
                        let call_start = Instant::now();
                        let resp = Python::with_gil(|py| -> PyResult<String> {
                            let f_b = f.bind(py);
                            let a_b = a.bind(py);
                            let k_b = k.as_ref().map(|kk| kk.bind(py));
                            let result = match &k_b {
                                Some(kk) => f_b.call(a_b, Some(kk))?,
                                None => f_b.call(a_b, None)?,
                            };
                            result.extract::<String>()
                        });
                        match resp {
                            Ok(s) => Ok((s, call_start.elapsed().as_secs_f64())),
                            Err(e) => Err(e),
                        }
                    }));
                }

                let mut outs = Vec::with_capacity(num_calls);
                let mut times = Vec::with_capacity(num_calls);
                for h in handles {
                    let r = h.join().map_err(|_| PyValueError::new_err("Thread panicked"))?;
                    let (s, t) = r?;
                    outs.push(s);
                    times.push(t);
                }
                Ok((outs, times))
            })?;

            // Seleziona il migliore con le distanze (CPU-only) fuori dal GIL
            let mbr_result = py.allow_threads(|| select_best_with_distances(&outputs))?;
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

// Funzione helper per creare il wrapper del decoratore majority voting.
fn create_majority_voting_wrapper(py: Python, func: PyObject, num_calls: usize) -> PyResult<PyObject> {
    let closure = PyCFunction::new_closure(py, None, None, move |args, kwargs| {
        Python::with_gil(|py| {
            let start_time = Instant::now();

            // Converte e clona args/kwargs/func in oggetti Py<...> trasferibili tra thread
            let args_owned = args.clone().unbind();
            let kwargs_owned = kwargs.map(|kw| kw.clone().unbind());
            let func_obj = func.clone_ref(py);

            // Prepara un vettore di specifiche di chiamata (uno per thread)
            let mut call_specs: Vec<(PyObject, Py<PyTuple>, Option<Py<PyDict>>)> = Vec::with_capacity(num_calls);
            for _ in 0..num_calls {
                let f_clone = func_obj.clone_ref(py);
                let a_clone = args_owned.clone_ref(py);
                let k_clone = kwargs_owned.as_ref().map(|kk| kk.clone_ref(py));
                call_specs.push((f_clone, a_clone, k_clone));
            }

            // Esegue le chiamate in parallelo con thread Rust, ciascuno acquisisce il GIL per la chiamata Python
            let (outputs, individual_times): (Vec<String>, Vec<f64>) = py.allow_threads(|| -> PyResult<(Vec<String>, Vec<f64>)> {
                let mut handles = Vec::with_capacity(num_calls);
                for (f, a, k) in call_specs.into_iter() {
                    handles.push(std::thread::spawn(move || {
                        let call_start = Instant::now();
                        // Esegue la singola chiamata al callable Python acquisendo il GIL nel thread
                        let resp = Python::with_gil(|py| -> PyResult<String> {
                            let f_b = f.bind(py);
                            let a_b = a.bind(py);
                            let k_b = k.as_ref().map(|kk| kk.bind(py));
                            let result = match &k_b {
                                Some(kk) => f_b.call(a_b, Some(kk))?,
                                None => f_b.call(a_b, None)?,
                            };
                            result.extract::<String>()
                        });
                        match resp {
                            Ok(s) => Ok((s, call_start.elapsed().as_secs_f64())),
                            Err(e) => Err(e),
                        }
                    }));
                }

                let mut outs = Vec::with_capacity(num_calls);
                let mut times = Vec::with_capacity(num_calls);
                for h in handles {
                    // Propaga errori di chiamata o panico del thread
                    let r = h.join().map_err(|_| PyValueError::new_err("Thread panicked"))?;
                    let (s, t) = r?;
                    outs.push(s);
                    times.push(t);
                }
                Ok((outs, times))
            })?;

            // Seleziona il migliore con majority voting (CPU-only) fuori dal GIL
            let voting_result = py.allow_threads(|| select_best_by_majority_voting(&outputs))?;
            let total_time = start_time.elapsed();

            // Crea il dizionario di output.
            let result_dict = PyDict::new(py);
            
            // Risposta selezionata
            result_dict.set_item("selected_response", PyString::new(py, &voting_result.best_response))?;
            
            // Tempo totale di esecuzione
            result_dict.set_item("total_execution_time", PyFloat::new(py, total_time.as_secs_f64()))?;
            
            // Tutte le risposte con i loro tempi
            let responses_list = PyList::empty(py);
            for (i, (response, time)) in outputs.iter().zip(individual_times.iter()).enumerate() {
                let response_dict = PyDict::new(py);
                response_dict.set_item("index", i)?;
                response_dict.set_item("response", PyString::new(py, response))?;
                response_dict.set_item("execution_time", PyFloat::new(py, *time))?;
                
                // Conta i voti per questa risposta
                let vote_count = voting_result.vote_counts.get(response).unwrap_or(&0);
                response_dict.set_item("vote_count", *vote_count)?;
                
                responses_list.append(response_dict)?;
            }
            result_dict.set_item("all_responses", responses_list)?;
            
            // Indice della risposta selezionata
            result_dict.set_item("selected_index", voting_result.best_index)?;
            
            // Conteggi dei voti
            let vote_counts_dict = PyDict::new(py);
            for (response, count) in &voting_result.vote_counts {
                vote_counts_dict.set_item(response, *count)?;
            }
            result_dict.set_item("vote_counts", vote_counts_dict)?;

            // Restituisci il dizionario.
            Ok::<PyObject, PyErr>(result_dict.into())
        })
    })?;
    Ok(closure.into())
}

// Il decoratore majority_voting.
// Gestisce sia @majority_voting (default num_calls=5) che @majority_voting(num_calls=N).
#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn majority_voting(py: Python, args: &Bound<PyTuple>, kwargs: Option<&Bound<PyDict>>) -> PyResult<PyObject> {
    let default_num_calls = 5;

    // Se chiamata senza args o con func come primo arg (caso @majority_voting).
    if args.len() == 1 && args.get_item(0)?.is_callable() {
        let func = args.get_item(0)?.clone().into();
        return create_majority_voting_wrapper(py, func, default_num_calls);
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
            create_majority_voting_wrapper(py, func, num_calls)
        })
    })?;
    Ok(decorator.into())
}

// Funzione helper per creare il wrapper del blender decorator
fn create_blender_wrapper(py: Python, func: PyObject, num_calls: usize, top_k: Option<usize>, inference_config: InferenceConfig) -> PyResult<PyObject> {
    let closure = PyCFunction::new_closure(py, None, None, move |args, kwargs| {
        Python::with_gil(|py| {
            let start_time = Instant::now();
            
            let mut outputs = Vec::new();
            let mut execution_times = Vec::new();
            
            // Esegui num_calls chiamate alla funzione
            for _ in 0..num_calls {
                let call_start = Instant::now();
                let result = func.call(py, args, kwargs)?;
                let call_duration = call_start.elapsed();
                execution_times.push(call_duration.as_secs_f64());
                
                let output_str = if let Ok(s) = result.extract::<String>(py) {
                    s
                } else {
                    result.call_method0(py, "str")?.extract::<String>(py)?
                };
                outputs.push(output_str);
            }
            
            // Usa sempre il blender AI-enhanced
            let blender_result = select_best_with_ai_blender(&outputs, top_k, &inference_config)?;
            let total_time = start_time.elapsed().as_secs_f64();
            
            // Crea il dizionario di output
            let result_dict = PyDict::new(py);
            result_dict.set_item("selected_response", &blender_result.best_response)?;
            result_dict.set_item("selected_index", blender_result.best_index)?;
            result_dict.set_item("total_execution_time", total_time)?;
            
            // Aggiungi i punteggi di ranking
            let ranking_scores: Vec<f64> = blender_result.ranking_scores;
            let py_ranking_scores = PyList::new(py, ranking_scores.clone())?;
            result_dict.set_item("ranking_scores", py_ranking_scores)?;
            
            // Aggiungi le candidate fuse
            let py_fused_candidates = PyList::new(py, &blender_result.fused_candidates)?;
            result_dict.set_item("fused_candidates", py_fused_candidates)?;
            
            // Aggiungi tutte le risposte con metadati
            let all_responses_list = PyList::empty(py);
            for (i, (response, exec_time)) in outputs.iter().zip(execution_times.iter()).enumerate() {
                let response_dict = PyDict::new(py);
                response_dict.set_item("response", response)?;
                response_dict.set_item("execution_time", exec_time)?;
                response_dict.set_item("index", i)?;
                response_dict.set_item("ranking_score", ranking_scores.get(i).unwrap_or(&0.0))?;
                all_responses_list.append(response_dict)?;
            }
            result_dict.set_item("all_responses", all_responses_list)?;
            
            // Il blender usa sempre AI enhancement
            result_dict.set_item("ai_enhanced", true)?;
            
            Ok::<PyObject, PyErr>(result_dict.into())
        })
    })?;
    Ok(closure.into())
}

// Decoratore blender che combina PairRanker e GenFuser
#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn blender(py: Python, args: &Bound<PyTuple>, kwargs: Option<&Bound<PyDict>>) -> PyResult<PyObject> {
    let default_num_calls = 5;
    let default_top_k = 3;
    
    // Caso 1: @blender (senza parametri) - ERRORE perché inference_config è obbligatorio
    if args.len() == 1 && args.get_item(0)?.is_callable() {
        return Err(PyValueError::new_err(
            "blender decorator requires 'inference_config' parameter. Usage: @blender(inference_config={...})"
        ));
    }
    
    // Caso 2: @blender(...) con parametri
    let mut num_calls = default_num_calls;
    let mut top_k = Some(default_top_k);
    let mut inference_config: Option<InferenceConfig> = None;
    
    if let Some(kw) = kwargs {
        if let Some(n) = kw.get_item("num_calls")? {
            num_calls = n.extract::<usize>()?;
        }
        if let Some(k) = kw.get_item("top_k")? {
            top_k = Some(k.extract::<usize>()?);
        }
        if let Some(config) = kw.get_item("inference_config")? {
            inference_config = Some(config.extract::<InferenceConfig>()?);
        }
    } else if args.len() >= 1 {
        num_calls = args.get_item(0)?.extract::<usize>()?;
        if args.len() >= 2 {
            top_k = Some(args.get_item(1)?.extract::<usize>()?);
        }
        if args.len() >= 3 {
            inference_config = Some(args.get_item(2)?.extract::<InferenceConfig>()?);
        }
    }
    
    // Verifica che inference_config sia stato fornito
    let inference_config = inference_config.ok_or_else(|| {
        PyValueError::new_err(
            "blender decorator requires 'inference_config' parameter. Usage: @blender(inference_config={'provider': '...', 'api_key': '...', 'model': '...'})"
        )
    })?;
    
    // Restituisci una closure che agisce come decoratore
    let decorator = PyCFunction::new_closure(py, None, None, move |dec_args, _dec_kwargs| {
        Python::with_gil(|py| {
            if dec_args.len() != 1 || !dec_args.get_item(0)?.is_callable() {
                return Err(PyValueError::new_err("Decorator expects a callable"));
            }
            let func = dec_args.get_item(0)?.clone().into();
            create_blender_wrapper(py, func, num_calls, top_k, inference_config.clone())
        })
    })?;
    Ok(decorator.into())
}

// Definizione del modulo Python.
// Passo 11: Registrazione delle funzioni decoratore nel modulo.
#[pymodule]
fn transforest(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(minimum_bayes_risk, m)?)?;
    m.add_function(wrap_pyfunction!(majority_voting, m)?)?;
    m.add_function(wrap_pyfunction!(blender, m)?)?;
    Ok(())
}