// Types for the classic-script build (dist/shaderintel.global.js), which defines a global `SI`.
//   <script src="node_modules/shaderintel/dist/shaderintel.global.js"></script>
import * as ShaderIntel from './shaderintel';

declare global {
  const SI: typeof ShaderIntel;
}

export {};
