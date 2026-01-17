#!/usr/bin/env python3
"""
Quick Start: Query2CAS Toolkit Demo

This script demonstrates the core functionality of the Query2CAS tool
within the utu-agent framework. It shows:

1. Tool initialization and registration
2. Basic molecule name to CAS number conversion
3. SMILES to CAS number conversion
4. Safety checking integration
5. Error handling

This is the simplest way to understand how Query2CAS works in the toolkit.
"""

import asyncio
import os
import sys


async def demo_query2cas_basic():
    """Demonstrate basic Query2CAS functionality."""
    print("\n" + "=" * 70)
    print("📋 Query2CAS Toolkit - Basic Demo")
    print("=" * 70)

    try:
        # Import the toolkit
        from utu.tools.chemcrow_toolkit import ChemcrowToolkit
        from utu.config import ToolkitConfig

        print("\n✅ ChemcrowToolkit imported successfully")

        # Initialize the toolkit with default config
        config = ToolkitConfig(
            name="chemcrow",
            mode="builtin",
            config={
                "temperature": 0.1,
                "llm_model": "gpt-3.5-turbo",
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "openai_api_base": os.getenv("OPENAI_API_BASE")
            },
        )
        toolkit = ChemcrowToolkit(config)
        print("✅ ChemcrowToolkit initialized")

        # Example 1: Query molecule by common name
        print("\n" + "-" * 70)
        print("Example 1: Get CAS number from molecule name")
        print("-" * 70)
        molecule_name = "aspirin"
        print(f"Query: {molecule_name}")

        result = await toolkit.query_molecule_cas(molecule_name)
        print(f"Result CAS: {result}")

        # Example 2: Get SMILES from name
        print("\n" + "-" * 70)
        print("Example 2: Convert molecule name to SMILES")
        print("-" * 70)
        print(f"Query: {molecule_name}")

        smiles = await toolkit.convert_name_to_smiles(molecule_name)
        print(f"SMILES: {smiles}")

        # Example 3: Get name from SMILES
        print("\n" + "-" * 70)
        print("Example 3: Convert SMILES to molecule name")
        print("-" * 70)
        # Use aspirin SMILES
        test_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        print(f"SMILES: {test_smiles}")

        name = await toolkit.convert_smiles_to_name(test_smiles)
        print(f"Name: {name}")

        # Example 4: Get molecular weight
        print("\n" + "-" * 70)
        print("Example 4: Get molecular weight from SMILES")
        print("-" * 70)
        print(f"SMILES: {test_smiles}")

        weight = await toolkit.get_molecular_weight(test_smiles)
        print(f"Molecular Weight: {weight}")

        # Example 5: Get functional groups
        print("\n" + "-" * 70)
        print("Example 5: Identify functional groups")
        print("-" * 70)
        print(f"SMILES: {test_smiles}")

        groups = await toolkit.get_functional_groups(test_smiles)
        print(f"Functional Groups: {groups}")

        # Example 6: Check patent status
        print("\n" + "-" * 70)
        print("Example 6: Check patent information")
        print("-" * 70)
        print(f"SMILES: {test_smiles}")

        patents = await toolkit.check_patent(test_smiles)
        print(f"Patent Info: {patents}")

        # Example 7: Check explosive status
        print("\n" + "-" * 70)
        print("Example 7: Check if molecule is explosive")
        print("-" * 70)
        print(f"SMILES: {test_smiles}")

        is_explosive = await toolkit.check_explosive(test_smiles)
        print(f"Explosive Status: {is_explosive}")

        # Example 8: Check controlled chemical status
        print("\n" + "-" * 70)
        print("Example 8: Check controlled chemical status")
        print("-" * 70)
        print(f"SMILES: {test_smiles}")

        controlled = await toolkit.check_controlled_chemical(test_smiles)
        print(f"Controlled Status: {controlled}")

        print("\n" + "=" * 70)
        print("✅ Demo completed successfully!")
        print("=" * 70 + "\n")

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n🔧 Make sure to install chemcrow:")
        print("   pip install -e /path/to/chemcrow")
        raise e
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


async def demo_agent_with_query2cas():
    """Demonstrate Query2CAS through the agent interface."""
    print("\n" + "=" * 70)
    print("🤖 Query2CAS via Agent - Interactive Demo")
    print("=" * 70)

    try:
        from utu.agents import get_agent
        from utu.config import ConfigLoader
        from utu.utils import PrintUtils

        # Load the chemcrow agent configuration
        config = ConfigLoader.load_agent_config("simple/chemcrow")
        agent = get_agent(config)

        print(f"\n✅ Agent initialized: {config.agent.name}")
        print("📚 This agent includes all ChemCrow tools including Query2CAS")

        # Run a sample query
        sample_query = "What is the CAS number for acetylsalicylic acid (aspirin)?"
        print(f"\n📝 Sample Query: {sample_query}")
        print("-" * 70)

        await agent.chat_streamed(sample_query)

        print("\n" + "-" * 70)
        print("\n✅ Query completed! You can now ask your own questions.")
        print("   Type 'exit' to quit.\n")

        # Interactive loop
        while True:
            user_input = await PrintUtils.async_print_input("🔬 Your query: ")
            if user_input.strip().lower() in ["exit", "quit", "q"]:
                break
            if user_input.strip():
                print("-" * 70)
                await agent.chat_streamed(user_input)
                print("-" * 70 + "\n")

    except FileNotFoundError:
        print("\n⚠️  Agent config not found. Please run the basic demo first.")
        await demo_query2cas_basic()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def print_usage():
    """Print usage information."""
    print("\n" + "=" * 70)
    print("Query2CAS Demo - Usage")
    print("=" * 70)
    print("\nUsage: python chemcrow_query2cas_demo.py [OPTIONS]")
    print("\nOptions:")
    print("  (no args)    Run basic toolkit demo showing all tools")
    print("  --agent      Run interactive agent with Query2CAS and other tools")
    print("  --help       Show this help message")
    print("\nEnvironment Variables:")
    print("  OPENAI_API_KEY: Required for agent mode")
    print("  SERP_API_KEY: Optional, for web search")
    print("  CHEMSPACE_API_KEY: Optional, for molecule pricing")
    print("\n" + "=" * 70 + "\n")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print_usage()
            return
        elif sys.argv[1] == "--agent":
            asyncio.run(demo_agent_with_query2cas())
            return

    # Default: run basic demo
    asyncio.run(demo_query2cas_basic())


if __name__ == "__main__":
    main()
