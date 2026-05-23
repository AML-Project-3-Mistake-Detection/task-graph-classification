# Using Annotation-Based Validation

Your `train.py` now supports validation against actual recipe annotations from `annotations/task_graphs/` instead of using only the embedded labels from the dataset.

## New Functions

### 1. `load_standard_task_graphs(annotations_dir='annotations/task_graphs')`
Loads all standard task graphs from the annotations directory.

```python
standard_graphs = load_standard_task_graphs()
# Returns: Dict[task_name: str, graph_data: dict]
#   - 'coffee': {'steps': {...}, 'edges': [...]}
#   - 'ramen': {'steps': {...}, 'edges': [...]}
#   - etc.
```

### 2. `compare_graphs_with_standard(observed_graph, standard_graph)`
Compares an observed execution graph against a standard recipe graph.

**Correctness criteria:**
- All observed edges must exist in the standard recipe's edge set
- Self-loops are ignored
- Returns `1` (correct) if valid, `0` (incorrect) if edges not found in standard

```python
label = compare_graphs_with_standard(observed_data, standard_graph)
# Returns: 0 or 1
```

### 3. `rebuild_validation_labels_from_annotations(dataset, standard_graphs, indices)`
Rebuilds validation labels by comparing all samples in a dataset against standard graphs.

```python
# Get annotation-based labels for validation set
val_labels = rebuild_validation_labels_from_annotations(
    dataset, 
    standard_graphs, 
    indices=val_dataset.indices  # From torch.utils.data.random_split
)
# Returns: List[0, 1, 1, 0, ...]
```

## Usage Example

### Option 1: Replace validation labels in `train_standard()`

```python
def train_standard(args, device):
    # ... existing code ...
    
    # Load standard graphs
    standard_graphs = load_standard_task_graphs('annotations/task_graphs')
    use_annotations = len(standard_graphs) > 0
    
    # ... load dataset, split into train/val ...
    
    # If using annotations, rebuild validation labels
    if use_annotations:
        val_labels_annotation = rebuild_validation_labels_from_annotations(
            dataset,
            standard_graphs,
            indices=val_dataset.indices
        )
        print(f"✓ Generated {len(val_labels_annotation)} annotation-based labels")
    
    # ... training loop ...
    
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_epoch(...)
        
        # Evaluate and collect predictions
        val_loss, val_acc, val_f1, val_auc, val_prec, val_rec, val_probs, _ = evaluate(...)
        
        # Compare predictions against annotations if available
        if use_annotations:
            val_labels_np = np.array(val_labels_annotation)
            # Now val_labels_np contains annotation-based ground truth
            annotation_f1 = f1_score(val_labels_np, all_preds[-len(val_labels_np):], 
                                     average='macro', zero_division=0)
            print(f"  Annotation-based F1: {annotation_f1:.4f}")
```

### Option 2: Compare both label sources

```python
# During validation, compare predictions against both sources
embedded_f1 = f1_score(embedded_labels, predictions, average='macro')
annotation_f1 = f1_score(annotation_labels, predictions, average='macro')

print(f"F1 vs Embedded Labels:    {embedded_f1:.4f}")
print(f"F1 vs Annotation Labels:  {annotation_f1:.4f}")
```

## Important Notes

### Metadata Preservation
The `rebuild_validation_labels_from_annotations()` function works best when your dataset preserves `task_name` attributes. Make sure your Data objects have this field:

```python
# In extract_graphs.py (already done)
data = Data(
    x=x_tensor,
    edge_index=edge_index,
    y=y_tensor,
    recording_id=recording_id,
    task_name=task_name  # ← Required for annotation-based validation
)
```

### Label Comparison
- **Embedded labels** (current): From `error_annotations.json` - whether someone marked the execution as erroneous
- **Annotation labels** (new): Whether the observed graph structure matches the standard recipe
- These may not always match! An execution could:
  - Have correct steps but be marked as error due to quality issues
  - Have incorrect steps but still marked as correct
  
This difference is useful for understanding model behavior.

### Correctness Criteria
The `compare_graphs_with_standard()` function uses a **structural matching** approach:
- ✅ CORRECT: All observed edges match standard recipe edges
- ❌ INCORRECT: Any observed edge not found in standard recipe
- ⚠️ EDGE CASES: Self-loops ignored, missing edges in observed graph treated as correct

If you need different matching criteria (e.g., exact step sequence, subsequence matching), modify the function accordingly.

## Next Steps

1. **Enable annotation-based validation** in your training loop
2. **Compare metrics** between embedded and annotation-based labels
3. **Adjust matching criteria** if needed (see `compare_graphs_with_standard()`)
4. **Log both sources** to understand label quality and model behavior

## Files Modified

- `train.py`: Added 4 new functions + automatic loading in `train_standard()`
- No other files modified
