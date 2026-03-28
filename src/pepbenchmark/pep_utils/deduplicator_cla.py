from typing import Any, List, Tuple, Union

import pandas as pd

from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)

class ClassificationDeduplicator:
    """
    Classification Data Deduplicator
    
    Supports single-object and multi-object grouping deduplication:
    - Single-object: deduplication based on a single column grouping
    - Multi-object: deduplication based on multiple column grouping (e.g., protein-peptide pairs)
    """
    
    def __init__(self, group_cols: Union[str, List[str]] = 'BILN', label_col: str = 'Label'):
        """
        Initialize the deduplicator
        
        Args:
            group_cols: Column name(s) for grouping, can be a single string or list of strings
            label_col: Label column name, default is 'Label'
        """
        # Convert to list format uniformly
        if isinstance(group_cols, str):
            self.group_cols = [group_cols]
        else:
            self.group_cols = group_cols
        self.label_col = label_col
    
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform deduplication on classification data
        
        Args:
            df: Input DataFrame, must contain group_cols columns, if contains label_col column then perform label conflict resolution
            
        Returns:
            pd.DataFrame: Deduplicated DataFrame
        """
        # Validate input data
        for col in self.group_cols:
            if col not in df.columns:
                raise ValueError(f"Grouping column '{col}' does not exist in DataFrame")
        
        # Check if label column exists
        has_label_col = self.label_col in df.columns
        
        if has_label_col:
            # Validate that labels are 0 and 1
            unique_labels = df[self.label_col].unique()
            if not set(unique_labels).issubset({0, 1}):
                raise ValueError(f"Label column must only contain 0 and 1, found: {unique_labels}")
        
        df = df.copy()
        
        # If no label column, perform simple deduplication keeping first
        if not has_label_col:
            logger.info(f"Label column '{self.label_col}' not found, performing simple deduplication (keep first)")
            if len(self.group_cols) == 1:
                # Single object grouping
                logger.info(f"Executing single object grouping deduplication, grouping column: {self.group_cols[0]}")
                df_dedup = df.drop_duplicates(subset=self.group_cols[0], keep='first')
            else:
                # Multi object grouping
                logger.info(f"Executing multi object grouping deduplication, grouping columns: {self.group_cols}")
                df_dedup = df.drop_duplicates(subset=self.group_cols, keep='first')
            
            # Print simple deduplication statistics
            self._print_simple_statistics(df, df_dedup)
            return df_dedup
        
        def resolve_conflict(group: pd.DataFrame) -> pd.Series:
            """Resolve conflicts when the same object has multiple labels"""
            label_counts = group[self.label_col].value_counts()
            
            # If only one label, return the first row directly
            if len(label_counts) == 1:
                return group.iloc[0]
            
            # If multiple labels, choose the one with the highest count
            majority_label = label_counts.idxmax()
            
            # If counts are equal, return None to indicate deletion of all samples for this sequence
            if len(label_counts) > 1 and label_counts.iloc[0] == label_counts.iloc[1]:
                return None
            
            # Find the first row with the majority label
            for _, row in group.iterrows():
                if row[self.label_col] == majority_label:
                    result = row.copy()
                    result[self.label_col] = majority_label
                    return result
        
        # Group by grouping columns
        if len(self.group_cols) == 1:
            # Single object grouping
            logger.info(f"Executing single object grouping deduplication, grouping column: {self.group_cols[0]}")
            grouped = df.groupby(self.group_cols[0])
        else:
            # Multi object grouping
            logger.info(f"Executing multi object grouping deduplication, grouping columns: {self.group_cols}")
            grouped = df.groupby(self.group_cols)
        
        # Apply conflict resolution strategy
        result_data = []
        processing_stats = {
            'total_groups': 0,
            'resolved_majority': 0,
            'removed_tie': 0,
            'single_label': 0
        }
        
        for group_key, group in grouped:
            processing_stats['total_groups'] += 1
            
            if len(group) == 1:
                # Only one sample, keep directly
                result_data.append(group.iloc[0].to_dict())
                processing_stats['single_label'] += 1
            else:
                # Multiple samples, need to resolve conflicts
                result = resolve_conflict(group)
                if result is not None:
                    result_data.append(result.to_dict())
                    processing_stats['resolved_majority'] += 1
                else:
                    processing_stats['removed_tie'] += 1
                    logger.warning(f"Group {group_key} has label tie, all samples deleted")
        
        # Create result DataFrame
        if result_data:
            df_dedup = pd.DataFrame(result_data)
        else:
            # If no results, create empty DataFrame but maintain column structure
            df_dedup = pd.DataFrame(columns=df.columns)
        
        # Print processing statistics
        self._print_statistics(df, df_dedup, processing_stats)
        
        return df_dedup
    
    def _print_simple_statistics(self, original_df: pd.DataFrame, deduped_df: pd.DataFrame) -> None:
        """Print simple deduplication statistics"""
        original_count = len(original_df)
        final_count = len(deduped_df)
        removed_count = original_count - final_count
        
        print(f"Simple deduplication statistics:")
        print(f"  Original data rows: {original_count}")
        print(f"  After deduplication rows: {final_count}")
        print(f"  Removed rows: {removed_count}")
        print(f"  Removal ratio: {removed_count/original_count*100:.2f}%")
        print(f"  Grouping columns: {self.group_cols}")
        print(f"  Deduplication strategy: Keep first duplicate")
    
    def _print_statistics(self, original_df: pd.DataFrame, deduped_df: pd.DataFrame, 
                         processing_stats: dict = None) -> None:
        """Print deduplication statistics"""
        original_count = len(original_df)
        final_count = len(deduped_df)
        removed_count = original_count - final_count
        
        print(f"Deduplication statistics:")
        print(f"  Original data rows: {original_count}")
        print(f"  After deduplication rows: {final_count}")
        print(f"  Removed rows: {removed_count}")
        print(f"  Removal ratio: {removed_count/original_count*100:.2f}%")
        print(f"  Grouping columns: {self.group_cols}")
        print(f"  Label column: {self.label_col}")
        
        if processing_stats:
            print(f"  Processing details:")
            print(f"    Total groups: {processing_stats['total_groups']}")
            print(f"    Single label groups: {processing_stats['single_label']}")
            print(f"    Majority label resolved: {processing_stats['resolved_majority']}")
            print(f"    Tie removed: {processing_stats['removed_tie']}")


# Usage example
if __name__ == "__main__":
    # Example usage

    df = pd.read_csv('combine.csv')
    deduplicator = ClassificationDeduplicator(group_cols=['sequence'], label_col='label')
    result = deduplicator.deduplicate(df)
    pass
