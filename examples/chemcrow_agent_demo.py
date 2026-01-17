#!/usr/bin/env python3
"""
ChemCrow Agent Demo - Interactive Chemistry Assistant

This demo shows how to use the ChemcrowToolkit within the utu-agent framework
to build an interactive chemistry assistant that can:
- Convert between molecule names, SMILES, and CAS numbers
- Analyze molecular properties
- Check safety and regulatory status
- Search chemical literature
- Plan chemical synthesis

Usage:
    python chemcrow_agent_demo.py

Environment Variables:
    OPENAI_API_KEY: Required for LLM functionality
    SERP_API_KEY: Optional, for web search capabilities
    CHEMSPACE_API_KEY: Optional, for molecule pricing
    RXN4CHEM_API_KEY: Optional, for reaction prediction
    SEMANTIC_SCHOLAR_API_KEY: Optional, for literature search

Example Queries:
    - "What is the CAS number for aspirin?"
    - "Convert ethanol to SMILES notation"
    - "What is the molecular weight of caffeine?"
    - "Is acetone an explosive?"
    - "Tell me about the synthesis of acetylsalicylic acid"
    - "Find papers about glucose metabolism"
"""

import asyncio
import sys

from utu.agents import get_agent
from utu.config import ConfigLoader
from utu.utils import AgentsUtils, PrintUtils


def print_welcome():
    """Print welcome message with tool information."""
    print("\n" + "=" * 70)
    print("🧪 ChemCrow Agent - Interactive Chemistry Assistant")
    print("=" * 70)
    print("\nAvailable Chemistry Tools:")
    print("  • Molecular Conversion: name ↔ SMILES ↔ CAS number")
    print("  • Property Analysis: molecular weight, functional groups, similarity")
    print("  • Safety Checking: explosive check, controlled chemical detection")
    print("  • Literature Search: scholarly paper search and analysis")
    print("  • Patent Information: patent status and IP investigation")
    print("  • Reaction Planning: synthesis prediction and retrosynthesis")
    print("\nExample Queries:")
    print("  - What is the CAS number for aspirin?")
    print("  - Convert ethanol to SMILES")
    print("  - What's the molecular weight of caffeine?")
    print("  - Is this molecule controlled? (with SMILES)")
    print("  - Find synthesis routes for acetylsalicylic acid")
    print("\nType 'exit', 'quit', or 'q' to quit")
    print("=" * 70 + "\n")


async def run_interactive_demo():
    """Run an interactive chemistry assistant session."""
    print_welcome()

    try:
        # Load agent configuration for chemcrow agent
        config = ConfigLoader.load_agent_config("simple/chemcrow")
        agent = get_agent(config)

        print(f"✅ Agent initialized: {config.agent.name}")
        print(f"📚 Loaded tools:")
        if hasattr(agent, 'tools') and agent.tools:
            for tool in agent.tools[:5]:  # Show first 5 tools
                print(f"   - {tool.name}")
            if len(agent.tools) > 5:
                print(f"   ... and {len(agent.tools) - 5} more tools")
        print()

        turn_id = 0
        while True:
            try:
                user_input = await PrintUtils.async_print_input("🔬 Query (or 'q' to quit): ")

                # Check for exit command
                if user_input.strip().lower() in ["exit", "quit", "q"]:
                    print("\n👋 Thank you for using ChemCrow Agent!")
                    break

                # Skip empty input
                if not user_input.strip():
                    print("⚠️  Please enter a query")
                    continue

                print("\n⏳ Processing your query...")
                print("-" * 70)

                # Send query to agent and stream response
                await agent.chat_streamed(user_input)

                print("\n" + "-" * 70)
                turn_id += 1
                print()

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error during query: {e}")
                print("   Please try again with a different query")
                continue

    except FileNotFoundError:
        print("❌ Error: Could not find agent configuration 'simple/chemcrow'")
        print("   Make sure the config file exists: configs/agents/simple/chemcrow.yaml")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        print("\n🔍 Troubleshooting:")
        print("  1. Check that OPENAI_API_KEY is set: echo $OPENAI_API_KEY")
        print("  2. Verify chemcrow is installed: pip install -e /path/to/chemcrow")
        print("  3. Check configuration files in configs/agents/simple/")
        sys.exit(1)


async def run_demo_examples():
    """Run predefined examples without interactive input."""
    print_welcome()

    example_queries = [
        "What is the CAS number for aspirin?",
        "Convert glucose to SMILES notation",
        "What's the molecular weight of caffeine?",
    ]

    try:
        config = ConfigLoader.load_agent_config("simple/chemcrow")
        agent = get_agent(config)

        print(f"✅ Agent initialized: {config.agent.name}\n")

        for i, query in enumerate(example_queries, 1):
            print(f"📝 Example {i}: {query}")
            print("-" * 70)

            try:
                await agent.chat_streamed(query)
                print("\n" + "-" * 70 + "\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    import os

    # Check if running interactively or with examples
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        asyncio.run(run_demo_examples())
    else:
        asyncio.run(run_interactive_demo())


if __name__ == "__main__":
    main()
