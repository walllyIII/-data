import pandas as pd
import numpy as np
import joblib
import json
import logging
from typing import Dict, List, Optional
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EndometriosisJSONPredictor:
    """Class for predicting endometriosis from JSON input"""

    def __init__(self,
                 model_path: str = os.path.join(os.path.dirname(__file__), "models", "rf_model_endometriosis.pkl"),
                 scaler_path: str = os.path.join(os.path.dirname(__file__), "models", "scaler_endometriosis.pkl")):
        """
        Initialize predictor with pre-trained model and scaler
        Args:
            model_path: Path to the pre-trained model
            scaler_path: Path to the pre-trained scaler
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features = [
            'ADAT1', 'FBXO45', 'PANX1', 'LOX', 'PPP3R1', 'CLSTN2', 'ELOVL6', 'COPG2', 'ZNF462', 'TIAM2',
            'RBBP5', 'TMEM17', 'MAGI1', 'PNMA2', 'CCNA1', 'EML1', 'USP46', 'AREL1', 'UTP20', 'LOC100507477',
            'SSX2IP', 'TRIM24', 'CEP76', 'BPGM', 'MAP3K20', 'MOSPD1', 'TEX9', 'LAMA1', 'CDC42BPA', 'CCDC113',
            'TMEM117', 'CEP128', 'TMEM206', 'ZBED6', 'NETO2', 'PCSK5', 'PPIG', 'TMA16', 'KIAA1211', 'RCAN1',
            'ERI1', 'PLK2', 'GALNT1', 'TXLNG', 'NCEH1', 'INPP5F', 'IFIT3', 'SHCBP1', 'WDR76', 'TTL',
            'CTSZ', 'SWAP70', 'MTFR2', 'GCNT1', 'LLPH', 'SLC27A6', 'KBTBD7', 'HMCN1', 'LGALSL', 'NAA15',
            'SCGB3A1', 'SGO2', 'ECT2', 'CADM1', 'RIT1', 'F13A1', 'MORC4', 'EGFL6', 'TSC22D4', 'SPAG1',
            'RPGRIP1L', 'PHTF1', 'ANKMY2', 'HPS5', 'ZFP90', 'DCDC2', 'LMNB1', 'PTGER3', 'PPP2R5E', 'RND3',
            'RAI14', 'ST6GAL2', 'C1QTNF7', 'PSD3', 'CEP83', 'TUNAR', 'RASGRP1', 'ZNF271P', 'FRY', 'AURKA',
            'SRGAP2C', 'CAPSL', 'EZH2', 'GRB14', 'BUB1', 'RGS7BP', 'MSRB3', 'JUNB', 'PLPPR4', 'BPIFB1',
            'MUC5B', 'EPB41L3', 'CDH2', 'CCNE2', 'UPF3B', 'RSPH1', 'C9orf135', 'NEDD9', 'IQCG', 'BAG2',
            'BRIP1', 'ETV1', 'CENPE', 'IFI44L', 'MSH2', 'ZNF23', 'PIP5K1B', 'CPM', 'EPHA5', 'ENPP5',
            'EIF4G3', 'SPATA18', 'ANK2', 'PAG1', 'OLFM4', 'LRRC3B', 'CA2', 'NUF2', 'PMAIP1', 'ARHGEF26',
            'VTCN1', 'SNTN', 'WIF1', 'GCLC', 'CDK1', 'ITGA9', 'MTUS2', 'ELN', 'TNFRSF19', 'FKBP8', 'FCGBP',
            'CDC20B', 'TFF3', 'KIF11', 'ANLN', 'DIO2', 'E2F7', 'FAM81B', 'FOS', 'APOD', 'PRC1', 'ADAM12',
            'APOBEC3B', 'NEFM', 'DLGAP5', 'PNMA8A', 'PPP2R2C', 'LINC00645', 'CCNB1', 'CENPU', 'LGR5',
            'CRISPLD1', 'CA4', 'RRM2', 'LTF', 'HSPB6', 'NPAS3', 'FOSB', 'ASPM', 'GALNT12', 'STXBP6',
            'CCDC146', 'PBK', 'CTSW', 'UBXN6', 'KMO', 'FXYD2', 'MMP7', 'CFD', 'OVGP1', 'ACKR1'
        ]
        self.labels = ['Control', 'Case']
        self.load_model()

    def load_model(self):
        """Load pre-trained model and scaler"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logger.info("Successfully loaded model and scaler")
        except Exception as e:
            logger.error(f"Failed to load model or scaler: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def validate_json_input(self, data: Dict) -> Optional[List[Dict]]:
        """
        Validate JSON input
        Args:
            data: JSON input dictionary
        Returns:
            List of validated sample dictionaries, or None
        """
        try:
            if not isinstance(data, dict) or 'samples' not in data:
                logger.error("JSON must have 'samples' key")
                return None

            samples = data['samples']
            if not isinstance(samples, list):
                logger.error("'samples' must be a list")
                return None

            validated_samples = []
            for sample in samples:
                if not isinstance(sample, dict) or 'sample_id' not in sample or 'gene_expression' not in sample:
                    logger.error("Sample missing 'sample_id' or 'gene_expression'")
                    return None

                sample_id = sample['sample_id']
                gene_expr = sample['gene_expression']
                if not isinstance(gene_expr, dict):
                    logger.error(f"Gene expression for {sample_id} must be a dictionary")
                    return None

                missing_genes = [g for g in self.features if g not in gene_expr]
                if missing_genes:
                    logger.error(f"Sample {sample_id} missing genes: {missing_genes}")
                    return None

                for gene in self.features:
                    if not isinstance(gene_expr[gene], (int, float)):
                        logger.error(f"Non-numeric value for {gene} in {sample_id}")
                        return None

                validated_samples.append({
                    'sample_id': sample_id,
                    'gene_expression': {g: float(gene_expr[g]) for g in self.features}
                })
            return validated_samples
        except Exception as e:
            logger.error(f"JSON validation failed: {str(e)}")
            return None

    def preprocess_data(self, samples: List[Dict]) -> Optional[pd.DataFrame]:
        """
        Preprocess gene expression data
        Args:
            samples: List of validated sample dictionaries
        Returns:
            Preprocessed feature matrix, or None
        """
        try:
            data = {
                s['sample_id']: [s['gene_expression'][g] for g in self.features]
                for s in samples
            }
            df = pd.DataFrame.from_dict(data, orient='index', columns=self.features)
            df.fillna(df.mean(), inplace=True)
            X_scaled = self.scaler.transform(df)
            return pd.DataFrame(X_scaled, columns=self.features, index=df.index)
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return None

    def predict(self, data: Dict) -> Dict:
        """
        Predict endometriosis from JSON data
        Args:
            data: JSON input dictionary
        Returns:
            Prediction results dictionary
        """
        try:
            # Validate input
            samples = self.validate_json_input(data)
            if samples is None:
                return {"status": "error", "message": "Invalid JSON input"}

            # Preprocess data
            X = self.preprocess_data(samples)
            if X is None:
                return {"status": "error", "message": "Data preprocessing failed"}

            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)

            # Format results
            results = {
                "status": "success",
                "predictions": [
                    {
                        "sample_id": X.index[i],
                        "prediction": self.labels[pred],
                        "probability_control": float(probabilities[i][0]),
                        "probability_case": float(probabilities[i][1])
                    }
                    for i, pred in enumerate(predictions)
                ]
            }
            return results
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {"status": "error", "message": str(e)}


# Pydantic model for input validation
class GeneExpressionSample(BaseModel):
    sample_id: str
    gene_expression: Dict[str, float]


class PredictionInput(BaseModel):
    samples: List[GeneExpressionSample]


app = FastAPI()

# CORS for JiuTian and Railway compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(input_data: PredictionInput):
    """
    Predict endometriosis from JSON input
    Args:
        input_data: JSON input with samples and gene expression data
    Returns:
        JSON response with predictions
    """
    try:
        # Convert Pydantic model to dict
        data = input_data.dict()
        predictor = EndometriosisJSONPredictor()
        result = predictor.predict(data)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        logger.error(f"API prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Use environment variable for port (Railway requirement)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
