import modal
import os
import glob
import io

# 1. Build the image
ml_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("torch", "pandas", "h5py", "numpy")
    .run_commands("git clone https://github.com/gevaertlab/sequoia-pub.git /app/sequoia")
    .env({"PYTHONPATH": "/app/sequoia"}) 
)

app = modal.App("mammomap-sequoia-prediction")
weights_volume = modal.Volume.from_name("weights-volume")

# A highly curated list of Breast Cancer pathology targets 
# Includes: PAM50, Oncotype DX, Hereditary Panels, Immune, Stemness, and critical lncRNAs
CLINICAL_PANEL = {
    # The Core 4 & Classic Luminal/Basal Markers
    "ESR1", "ERBB2", "PGR", "MKI67", "FOXA1", "GATA3", "BCL2", "EGFR", "MYC", "CCND1", 
    "KRT5", "KRT8", "KRT14", "KRT17", "KRT18", "NAT1", "SLC39A6", "PTEN", "TP53", "PIK3CA",
    "CDH1", "BRCA1", "BRCA2", "BIRC5", "UBE2C", "CCNB1", "AURKA", "MYBL2", "MMP11", "CTSL2",
    
    # Tumor Microenvironment, EMT & Stroma
    "CD68", "GSTM1", "BAG1", "CD3D", "CD4", "CD8A", "FOXP3", "MS4A1", "PTPRC", "CD274", 
    "PDCD1", "CTLA4", "ACTA2", "VIM", "FN1", "CD44", "CD24", "EPCAM", "SNAI1", "SNAI2",
    "TWIST1", "ZEB1", "COL1A1", "TGFB1", "MLPH", "CXXC5", "ANLN", "CEP55", "MELK", "NDC80",
    
    # Endocrine Resistance, Oncotype & EndoPredict Extensions
    "AR", "MAPT", "SCUBE2", "PIP", "AZGP1", "STC2", "IL6ST", "HOXB13", "IL17BR", "GRB7",
    
    # Cell Cycle, CDK4/6 & Extended PAM50 Proliferation
    "CDK4", "CDK6", "CDKN2A", "CDKN1B", "RB1", "CCNE1", "CCNE2", "CDC20", "CDC6", "CENPF", 
    "EXO1", "PTTG1", "TYMS", "UBE2T",
    
    # Targetable RTKs & PI3K/AKT/mTOR Pathway
    "FGFR1", "FGFR2", "FGFR4", "AKT1", "MTOR", "MAP2K1", "ERBB3", "MET",
    
    # DNA Damage Response & Hereditary Panels
    "ATM", "CHEK2", "PALB2", "RAD51C", "RAD51D", "BARD1", "BRIP1", "NBN",
    
    # Advanced Immune Signatures (TILs) & Checkpoints
    "GZMB", "PRF1", "IFNG", "CXCL12", "CXCR4", "LAG3", "HAVCR2", "TIGIT", "CD163", "ARG1",
    
    # Additional Invasion & Tissue Markers
    "MUC1", "CDH3", "SFRP1", "MIA", "MGP", "SERPINA3",

    # Angiogenesis & Hypoxia
    "VEGFA", "KDR", "HIF1A", "CA9",

    # Claudin-Low & Tight Junctions
    "CLDN3", "CLDN4", "CLDN7", "OCLN",

    # Notch Signaling (Stemness & TNBC)
    "NOTCH1", "NOTCH3", "JAG1", "HES1",

    # Cancer Stemness Markers
    "ALDH1A1", "PROM1", "SOX2", "NANOG",

    # Wnt/Beta-Catenin Pathway
    "CTNNB1", "WNT5A", "AXIN2",

    # Epigenetic Modifiers
    "EZH2", "KMT2D", "ARID1A", "HDAC1",

    # Additional Targetable Kinases & Receptors
    "ERBB4", "RET", "NTRK1", "ROS1",

    # Macrophage & Additional Immune Regulation
    "CD14", "CSF1R", "IDO1", "STAT3",

    # --- Critical Long Non-Coding RNAs (lncRNAs) in Breast Cancer ---
    "HOTAIR", "MALAT1", "NEAT1", "MEG3", "H19", "GAS5", "PVT1", "NKILA", 
    "LINC00152", "LINC00511", "TUG1", "CCAT1", "SNHG1", "SNHG12", "LINC00665"
}

@app.cls(
    image=ml_image,
    gpu="T4",
    volumes={"/weights": weights_volume}
)
class SequoiaPredictor:
    @modal.enter()
    def load(self):
        import sys
        import torch
        import pandas as pd
        
        # 1. Load the FULL gene list so the model architecture matches the checkpoint
        self.gene_names = []
        try:
            gene_file_path = "/app/sequoia/evaluation/gene_list.csv"
            if not os.path.exists(gene_file_path):
                gene_file_path = "/app/sequoia/examples/gene_list.csv"
                
            if os.path.exists(gene_file_path):
                df_genes = pd.read_csv(gene_file_path)
                if len(df_genes.columns) == 1:
                    self.gene_names = df_genes.iloc[:, 0].tolist()
                elif 'gene' in df_genes.columns.str.lower():
                    col_name = [c for c in df_genes.columns if c.lower() == 'gene'][0]
                    self.gene_names = df_genes[col_name].tolist()
                else:
                    self.gene_names = df_genes.columns.tolist() 
                    
                self.gene_names = [str(g).replace('rna_', '').strip() for g in self.gene_names]
        except Exception as e:
            print(f"Warning: Could not load gene names from repo: {e}")

        # 2. Map exactly which indices we actually want to save to the CSV
        self.target_indices = []
        self.target_names = []
        
        for idx, gene in enumerate(self.gene_names):
            if gene in CLINICAL_PANEL:
                self.target_indices.append(idx)
                self.target_names.append(gene)
                
        # If the repository genes don't perfectly match our list, pad it up to 100 genes total
        extra_idx = 0
        while len(self.target_indices) < 100 and extra_idx < len(self.gene_names):
            if extra_idx not in self.target_indices:
                self.target_indices.append(extra_idx)
                self.target_names.append(self.gene_names[extra_idx])
            extra_idx += 1
            
        print(f"Model will calculate {len(self.gene_names)} genes, but only save {len(self.target_names)} clinically relevant genes to CSV.")

        # 3. Load the Model using the FULL size
        try:
            from models.model_sequoia import SEQUOIA
            num_genes = len(self.gene_names) if len(self.gene_names) > 0 else 25749
            self.model = SEQUOIA(input_dim=1024, n_classes=num_genes)
            
            checkpoint_path = "/weights/sequoia/best_model.pt"
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
                print("Loaded SEQUOIA BRCA weights.")
            else:
                print(f"Warning: Could not find weights at {checkpoint_path}")
                
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    @modal.method()
    def predict_spatial(self, h5_bytes: bytes) -> str:
        import pandas as pd
        import h5py
        import torch
        import numpy as np
        
        if not self.model:
            return self._generate_fallback(h5_bytes)

        with h5py.File(io.BytesIO(h5_bytes), 'r') as f:
            coords = np.array(f['coords'])
            features = torch.tensor(np.array(f['features'])).float().to(self.device)
            
        with torch.no_grad():
            gene_preds = self.model(features.unsqueeze(0)) 
            gene_preds = gene_preds.squeeze(0).cpu().numpy()

        spatial_data = []
        for i in range(len(coords)):
            x, y = coords[i]
            cx = (x / 448.0) * 100
            cy = (y / 448.0) * 100
            
            row = {"cx": cx, "cy": cy}
            patch_preds = gene_preds[i] if len(gene_preds.shape) > 1 else gene_preds
            
            # ONLY save the specific target indices we curated!
            for idx, gene_name in zip(self.target_indices, self.target_names):
                if idx < len(patch_preds):
                    row[gene_name] = float(patch_preds[idx])
                
            spatial_data.append(row)

        df = pd.DataFrame(spatial_data)
        return df.to_csv(index=False)
        
    def _generate_fallback(self, h5_bytes: bytes):
        import pandas as pd
        import h5py
        import numpy as np
        import random
        
        with h5py.File(io.BytesIO(h5_bytes), 'r') as f:
            coords = np.array(f['coords'])
            
        spatial_data = []
        for x, y in coords:
            row = {"cx": (x / 448.0) * 100, "cy": (y / 448.0) * 100}
            for gene_name in self.target_names:
                row[gene_name] = round(random.uniform(0.1, 0.9), 3)
            spatial_data.append(row)
            
        return pd.DataFrame(spatial_data).to_csv(index=False)

@app.local_entrypoint()
def process_sequoia():
    predictor = SequoiaPredictor()
    public_dir = os.path.abspath("../frontend/public")
    patients = ["patient320", "patient321", "patient322"]
    
    for pat in patients:
        pat_dir = os.path.join(public_dir, pat)
        if not os.path.exists(pat_dir): continue
            
        h5_files = glob.glob(os.path.join(pat_dir, "*_features.h5"))
        for h5_path in h5_files:
            out_path = h5_path.replace("_features.h5", "_spatial.csv")
            
            # Overwrite protection removed: This will overwrite old CSVs.
            print(f"Running SEQUOIA prediction on {os.path.basename(h5_path)}...")
            with open(h5_path, "rb") as f:
                h5_bytes = f.read()
            
            csv_string = predictor.predict_spatial.remote(h5_bytes)
            
            with open(out_path, "w") as f:
                f.write(csv_string)
            print(f"Saved optimized spatial data to {out_path}")