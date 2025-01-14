# src/quantum_system/language.py

from enum import Enum
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class LanguageLevel(Enum):
    """
    Represents different levels of the quantum programming language hierarchy,
    as described in the research papers. Examples include:
      - PROGRAMMING: High-level language (like Q# or a Python DSL).
      - EXECUTABLE: A lower-level representation of quantum operations (gate sets).
      - ISA: Instruction Set Architecture (machine-specific instructions).
      - OPCODE: Numeric codes identifying hardware instructions or micro-ops.
      - GATES: Raw fields specifying analog gate pulses or waveforms.
    """
    PROGRAMMING = "programming_language"
    EXECUTABLE = "program_executable"
    ISA = "instruction_set_architecture"
    OPCODE = "opcode"
    GATES = "gates_fields"


@dataclass
class Instruction:
    """
    Represents a single quantum instruction at a particular language level.
    
    Attributes:
        level: The language level at which this instruction is defined.
        operation: Name or identifier of the quantum operation (e.g., 'H', 'CNOT', 'M').
        parameters: Additional data needed by the operation (angles, durations, etc.).
        qubits: The list of qubits on which this instruction acts.
    """
    level: LanguageLevel
    operation: str
    parameters: Dict[str, Union[float, int, str]] = None
    qubits: List[int] = None


class QuantumCompiler:
    """
    A compiler that translates quantum instructions between language levels.
    This includes parsing a high-level quantum program, generating gate-level
    instructions, or mapping to hardware-specific codes.
    """

    def __init__(self):
        """
        Initialize with sample or known mappings between levels. 
        In a real system, you would load these from a config or database.
        """
        self.supported_operations = {
            LanguageLevel.PROGRAMMING:    ['h', 'cnot', 'measure'],
            LanguageLevel.EXECUTABLE:     ['HADAMARD', 'CNOT', 'MEASURE'],
            LanguageLevel.ISA:            ['H', 'CX', 'M'],
            LanguageLevel.OPCODE:         ['0x01', '0x02', '0x03'],
            LanguageLevel.GATES:          ['pulse_h', 'pulse_cx', 'pulse_m'],
        }

    def compile_program(self, 
                        source_code: str, 
                        source_level: LanguageLevel,
                        target_level: LanguageLevel) -> List[Instruction]:
        """
        Compile a quantum program from one language level to another.

        Args:
            source_code: String representing the program or code snippet
                         at the `source_level`.
            source_level: The language level of the input code.
            target_level: The desired output language level.

        Returns:
            A list of Instruction objects at the `target_level`.
        """
        logger.info(f"Compiling program from {source_level.name} to {target_level.name}.")
        instructions = self._parse_source(source_code, source_level)
        return self._translate_instructions(instructions, source_level, target_level)

    def _parse_source(self, 
                      source_code: str, 
                      level: LanguageLevel) -> List[Instruction]:
        """
        Parse the source code into a list of instructions based on the level's syntax.

        NOTE: This is highly simplified. Real implementations would involve
        tokenizers, parsers, and AST transformations.

        Returns:
            A list of Instruction objects at the `level`.
        """
        logger.debug(f"Parsing source code at level {level.name}:\n{source_code}")
        # Mock parse: assume each line is "op qubit_list"
        instructions = []
        lines = source_code.strip().split("\n")
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            op = parts[0].lower()
            qubit_list = [int(q) for q in parts[1:]] if len(parts) > 1 else []
            if op in self.supported_operations.get(level, []):
                instructions.append(Instruction(
                    level=level,
                    operation=op,
                    qubits=qubit_list,
                    parameters={}
                ))
            else:
                logger.warning(f"Operation '{op}' not recognized at level {level.name}.")
        return instructions

    def _translate_instructions(self,
                                instructions: List[Instruction],
                                source_level: LanguageLevel,
                                target_level: LanguageLevel) -> List[Instruction]:
        """
        Translate instructions from the source language level to the target level.

        For each instruction, find the equivalent operation name at the target level.
        If no direct mapping is found, logs a warning or omits the instruction.

        Returns:
            A list of Instruction objects at the target level.
        """
        logger.debug(f"Translating {len(instructions)} instructions from {source_level.name} to {target_level.name}.")

        source_ops = self.supported_operations.get(source_level, [])
        target_ops = self.supported_operations.get(target_level, [])

        translated_instructions = []
        for instr in instructions:
            if instr.operation in source_ops:
                # Find the index of the operation in the source ops
                op_index = source_ops.index(instr.operation)
                # Attempt to map it to the target op at the same index
                if op_index < len(target_ops):
                    target_op = target_ops[op_index]
                    translated_instructions.append(Instruction(
                        level=target_level,
                        operation=target_op,
                        qubits=instr.qubits,
                        parameters=instr.parameters
                    ))
                else:
                    logger.warning(f"No direct mapping for operation {instr.operation} at target level {target_level.name}.")
            else:
                logger.warning(f"Operation {instr.operation} not found in source ops for {source_level.name}.")

        return translated_instructions


class InstructionSetArchitecture:
    """
    Represents a machine-level ISA for quantum operations, specifying formats,
    cycles, durations, and potentially hardware constraints.
    """

    def __init__(self):
        """
        Initialize the ISA with known instructions and their metadata. 
        In a real system, load from hardware specs or config files.
        """
        self.instruction_set = {
            'H': {
                'format': 'single_qubit',
                'cycles': 1,
                'duration_ns': 40,
            },
            'CX': {
                'format': 'two_qubit',
                'cycles': 2,
                'duration_ns': 100,
            },
            'M': {
                'format': 'measurement',
                'cycles': 4,
                'duration_ns': 300,
            }
        }
        logger.debug("ISA initialized with default instructions: H, CX, M.")

    def validate_instruction(self, instruction: Instruction) -> bool:
        """
        Check if a given instruction conforms to this ISA's specification.

        Args:
            instruction: An Instruction at the ISA level.

        Returns:
            True if valid, False otherwise.
        """
        if instruction.operation not in self.instruction_set:
            logger.warning(f"Operation '{instruction.operation}' not recognized in this ISA.")
            return False

        spec = self.instruction_set[instruction.operation]
        required_format = spec['format']

        if required_format == 'single_qubit' and len(instruction.qubits) != 1:
            logger.warning(f"Instruction '{instruction.operation}' requires 1 qubit.")
            return False
        elif required_format == 'two_qubit' and len(instruction.qubits) != 2:
            logger.warning(f"Instruction '{instruction.operation}' requires 2 qubits.")
            return False
        elif required_format == 'measurement' and (len(instruction.qubits) < 1):
            logger.warning(f"Measurement '{instruction.operation}' requires at least 1 qubit.")
            return False

        logger.debug(f"Instruction '{instruction.operation}' is valid under the current ISA.")
        return True
