"""ChemCrow Toolkit for chemical tools integration.

This toolkit wraps the chemcrow tools (Query2CAS, Query2SMILES, SMILES2Name, etc.)
into the utu-agent toolkit framework, allowing chemical queries to be used within agents.

Required environment variables:
- OPENAI_API_KEY: For LLM-dependent tools like SafetySummary
- SERP_API_KEY (optional): For WebSearch tool
- CHEMSPACE_API_KEY (optional): For molecule pricing
- RXN4CHEM_API_KEY (optional): For reaction prediction
- SEMANTIC_SCHOLAR_API_KEY (optional): For literature search
"""

import os
from typing import Optional

from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI

from ..config import ToolkitConfig
from .base import AsyncBaseToolkit, register_tool


class ChemcrowToolkit(AsyncBaseToolkit):
    """
    ChemcrowToolkit integrates chemical analysis tools for molecule identification and analysis.

    Attributes:
        config (ToolkitConfig): Configuration for the toolkit
        llm (BaseLanguageModel): Language model for tools that require LLM
        api_keys (dict): API keys for external services
    """

    def __init__(self, config: ToolkitConfig | dict = None) -> None:
        """Initialize ChemcrowToolkit with configuration and dependencies."""
        super().__init__(config)

        # Initialize LLM for tools that require it
        openai_api_key = self.config.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        llm_model = self.config.config.get("llm_model", "gpt-3.5-turbo")
        openai_api_base = self.config.config.get("openai_api_base", None)

        self.llm = ChatOpenAI(
            model_name=llm_model,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            temperature=self.config.config.get("temperature", 0.1),
        )

        # Prepare API keys from config or environment
        self.api_keys = {
            "OPENAI_API_KEY": openai_api_key,
            "SERP_API_KEY": self.config.config.get("serp_api_key") or os.getenv("SERP_API_KEY"),
            "CHEMSPACE_API_KEY": self.config.config.get("chemspace_api_key")
            or os.getenv("CHEMSPACE_API_KEY"),
            "RXN4CHEM_API_KEY": self.config.config.get("rxn4chem_api_key")
            or os.getenv("RXN4CHEM_API_KEY"),
            "SEMANTIC_SCHOLAR_API_KEY": self.config.config.get("semantic_scholar_api_key")
            or os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        }

        # Import and initialize chemcrow tools
        from chemcrow.tools import (
                ControlChemCheck,
                ExplosiveCheck,
                FuncGroups,
                GetMoleculePrice,
                MolSimilarity,
                PatentCheck,
                Query2CAS,
                Query2SMILES,
                RXNPredict,
                RXNPredictLocal,
                RXNRetrosynthesis,
                RXNRetrosynthesisLocal,
                SafetySummary,
                Scholar2ResultLLM,
                SMILES2Name,
                SMILES2Weight,
                SimilarControlChemCheck,
                WebSearch,
            )

        # Initialize core tools (no dependencies)
        self.query2cas_tool = Query2CAS()
        self.query2smiles_tool = Query2SMILES(self.api_keys.get("CHEMSPACE_API_KEY"))
        self.smiles2name_tool = SMILES2Name()
        self.mol_similarity_tool = MolSimilarity()
        self.smiles2weight_tool = SMILES2Weight()
        self.func_groups_tool = FuncGroups()
        self.patent_check_tool = PatentCheck()
        self.explosive_check_tool = ExplosiveCheck()
        self.control_chem_check_tool = ControlChemCheck()
        self.similar_control_chem_check_tool = SimilarControlChemCheck()

        # Initialize tools requiring LLM
        self.safety_summary_tool = SafetySummary(llm=self.llm)
        self.scholar_tool = Scholar2ResultLLM(
            llm=self.llm,
            openai_api_key=self.api_keys.get("OPENAI_API_KEY"),
            semantic_scholar_api_key=self.api_keys.get("SEMANTIC_SCHOLAR_API_KEY"),
        )

        # Initialize optional tools
        if self.api_keys.get("CHEMSPACE_API_KEY"):
            self.molecule_price_tool = GetMoleculePrice(self.api_keys.get("CHEMSPACE_API_KEY"))
        else:
            self.molecule_price_tool = None

        if self.api_keys.get("SERP_API_KEY"):
            self.web_search_tool = WebSearch(self.api_keys.get("SERP_API_KEY"))
        else:
            self.web_search_tool = None

        # Initialize reaction tools
        local_rxn = self.config.config.get("local_rxn", False)
        if local_rxn:
            self.rxn_predict_tool = RXNPredictLocal()
            self.rxn_retro_tool = RXNRetrosynthesisLocal()
        else:
            if self.api_keys.get("RXN4CHEM_API_KEY"):
                self.rxn_predict_tool = RXNPredict(self.api_keys.get("RXN4CHEM_API_KEY"))
                self.rxn_retro_tool = RXNRetrosynthesis(
                    self.api_keys.get("RXN4CHEM_API_KEY"),
                    self.api_keys.get("OPENAI_API_KEY"),
                )
            else:
                self.rxn_predict_tool = None
                self.rxn_retro_tool = None

    @register_tool
    async def query_molecule_cas(self, molecule_query: str) -> str:
        """Query CAS number from molecule name or SMILES.

        This tool converts a molecule name or SMILES string to its CAS number.
        It also checks if the molecule has any controlled chemical warnings.

        Args:
            molecule_query (str): The molecule name or SMILES string to query.
                                 Example: "aspirin" or "CC(=O)Oc1ccccc1C(=O)O"

        Returns:
            str: The CAS number of the molecule, or an error message if not found.
                 May include warnings about controlled chemicals.

        Example:
            >>> await toolkit.query_molecule_cas("aspirin")
            '50-78-2'
        """
        return self.query2cas_tool._run(molecule_query)

    @register_tool
    async def convert_name_to_smiles(self, molecule_name: str) -> str:
        """Convert molecule name to SMILES notation.

        This tool takes a molecule's common name and returns its SMILES representation.
        Uses PubChem database or ChemSpace API for conversion.

        Args:
            molecule_name (str): The common name of the molecule.
                                Example: "aspirin", "ethanol", "glucose"

        Returns:
            str: The SMILES string representation of the molecule.
                 May include controlled chemical warnings.

        Example:
            >>> await toolkit.convert_name_to_smiles("aspirin")
            'CC(=O)Oc1ccccc1C(=O)O'
        """
        return self.query2smiles_tool._run(molecule_name)

    @register_tool
    async def convert_smiles_to_name(self, smiles: str) -> str:
        """Convert SMILES notation to molecule name.

        This tool takes a SMILES string and returns the IUPAC or common name.

        Args:
            smiles (str): The SMILES string of the molecule.
                         Example: "CC(=O)Oc1ccccc1C(=O)O"

        Returns:
            str: The name of the molecule. May include controlled chemical warnings.

        Example:
            >>> await toolkit.convert_smiles_to_name("CC(=O)Oc1ccccc1C(=O)O")
            '2-acetoxybenzoic acid' or similar
        """
        return self.smiles2name_tool._run(smiles)

    @register_tool
    async def get_molecular_weight(self, smiles: str) -> str:
        """Get the molecular weight of a molecule from SMILES.

        Args:
            smiles (str): The SMILES string of the molecule.

        Returns:
            str: The molecular weight of the molecule.

        Example:
            >>> await toolkit.get_molecular_weight("CC(=O)Oc1ccccc1C(=O)O")
            '180.157 g/mol'
        """
        result = self.smiles2weight_tool._run(smiles)
        return str(result) if not isinstance(result, str) else result

    @register_tool
    async def get_functional_groups(self, smiles: str) -> str:
        """Identify functional groups in a molecule from SMILES.

        Args:
            smiles (str): The SMILES string of the molecule.

        Returns:
            str: Description of functional groups present in the molecule.

        Example:
            >>> await toolkit.get_functional_groups("CC(=O)Oc1ccccc1C(=O)O")
            'carboxylic acid, ester, benzene ring'
        """
        return self.func_groups_tool._run(smiles)

    @register_tool
    async def check_molecule_similarity(self, smiles1: str, smiles2: str) -> str:
        """Check similarity between two molecules.

        Args:
            smiles1 (str): SMILES string of the first molecule.
            smiles2 (str): SMILES string of the second molecule.

        Returns:
            str: Similarity score and description.
        """
        return self.mol_similarity_tool._run(f"{smiles1} {smiles2}")

    @register_tool
    async def check_patent(self, molecule_smiles: str) -> str:
        """Check if a molecule has any associated patents.

        Args:
            molecule_smiles (str): SMILES string of the molecule to check.

        Returns:
            str: Information about patents for this molecule.
        """
        return self.patent_check_tool._run(molecule_smiles)

    @register_tool
    async def check_explosive(self, molecule_smiles: str) -> str:
        """Check if a molecule is classified as explosive.

        Args:
            molecule_smiles (str): SMILES string of the molecule to check.

        Returns:
            str: Explosive classification result.
        """
        return self.explosive_check_tool._run(molecule_smiles)

    @register_tool
    async def check_controlled_chemical(self, molecule_smiles: str) -> str:
        """Check if a molecule is a controlled chemical.

        Args:
            molecule_smiles (str): SMILES string of the molecule to check.

        Returns:
            str: Controlled chemical status and any regulations.
        """
        return self.control_chem_check_tool._run(molecule_smiles)

    @register_tool
    async def check_similar_controlled_chemicals(self, molecule_smiles: str) -> str:
        """Find similar controlled chemicals to a given molecule.

        Args:
            molecule_smiles (str): SMILES string of the molecule to check.

        Returns:
            str: List of similar controlled chemicals.
        """
        return self.similar_control_chem_check_tool._run(molecule_smiles)

    @register_tool
    async def get_safety_summary(self, molecule_query: str) -> str:
        """Get comprehensive safety information for a molecule.

        Args:
            molecule_query (str): The molecule name or SMILES string.

        Returns:
            str: Safety summary including hazards and handling information.
        """
        return self.safety_summary_tool._run(molecule_query)

    @register_tool
    async def search_scholarly_literature(self, query: str) -> str:
        """Search for scholarly literature about a chemical topic.

        Uses Semantic Scholar API and PaperQA for comprehensive literature search.

        Args:
            query (str): The chemical question or topic to research.

        Returns:
            str: Answer synthesized from scholarly papers.
        """
        return self.scholar_tool._run(query)

    @register_tool
    async def web_search_chemical(self, query: str) -> str:
        """Search the web for chemical information.

        Args:
            query (str): The search query about chemicals.

        Returns:
            str: Web search results.
        """
        if self.web_search_tool is None:
            return "Web search tool not available. Set SERP_API_KEY environment variable."
        return self.web_search_tool._run(query)

    @register_tool
    async def get_molecule_price(self, molecule_smiles: str) -> str:
        """Get the price of a molecule from ChemSpace catalog.

        Args:
            molecule_smiles (str): SMILES string of the molecule.

        Returns:
            str: Pricing information from ChemSpace.
        """
        if self.molecule_price_tool is None:
            return "Molecule price lookup not available. Set CHEMSPACE_API_KEY environment variable."
        return self.molecule_price_tool._run(molecule_smiles)

    @register_tool
    async def predict_reaction(self, reactants_smiles: str) -> str:
        """Predict products of a chemical reaction.

        Args:
            reactants_smiles (str): SMILES strings of reactants, space-separated.

        Returns:
            str: Predicted products and reaction information.
        """
        if self.rxn_predict_tool is None:
            return (
                "Reaction prediction tool not available. "
                "Set RXN4CHEM_API_KEY environment variable or enable local_rxn."
            )
        return self.rxn_predict_tool._run(reactants_smiles)

    @register_tool
    async def retrosynthesis_planning(self, target_smiles: str) -> str:
        """Plan retrosynthesis pathways for a target molecule.

        Args:
            target_smiles (str): SMILES string of the target molecule.

        Returns:
            str: Retrosynthesis routes and available commercial precursors.
        """
        if self.rxn_retro_tool is None:
            return (
                "Retrosynthesis tool not available. "
                "Set RXN4CHEM_API_KEY environment variable or enable local_rxn."
            )
        return self.rxn_retro_tool._run(target_smiles)
